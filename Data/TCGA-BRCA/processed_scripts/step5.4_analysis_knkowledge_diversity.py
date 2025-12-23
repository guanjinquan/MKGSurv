import json
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
import re

# 首次运行需要下载 nltk 数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DiversityEvaluator:
    def __init__(self, texts):
        """
        :param texts: list of strings (raw text data)
        """
        self.raw_texts = texts
        # 简单的分词处理：转小写，移除标点
        self.tokenized_texts = [self._tokenize(t) for t in texts if t.strip()]
        self.num_docs = len(self.tokenized_texts)

    def _tokenize(self, text):
        text = text.lower()
        # 移除非字母数字字符，仅保留空格
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return nltk.word_tokenize(text)

    def vocab_size(self):
        """
        计算词汇表大小 (Vocabulary Size)。
        统计所有文本中出现的不重复单词总数。
        """
        if self.num_docs == 0: return 0
        
        unique_tokens = set()
        for tokens in self.tokenized_texts:
            unique_tokens.update(tokens)
            
        return len(unique_tokens)

    def distinct_n(self, n=1):
        """
        计算 Distinct-N 指标。
        Distinct-N = unique_ngrams / total_ngrams
        """
        if self.num_docs == 0: return 0.0
        
        all_ngrams = []
        for tokens in self.tokenized_texts:
            all_ngrams.extend(list(ngrams(tokens, n)))
            
        if len(all_ngrams) == 0:
            return 0.0
            
        return len(set(all_ngrams)) / len(all_ngrams)

    def entropy_n(self, n=2):
        """
        计算 N-gram Entropy。
        衡量 N-gram 分布的不可预测性。
        """
        if self.num_docs == 0: return 0.0
        
        all_ngrams = []
        for tokens in self.tokenized_texts:
            all_ngrams.extend(list(ngrams(tokens, n)))
            
        if not all_ngrams:
            return 0.0
            
        freq_dist = Counter(all_ngrams)
        total_count = sum(freq_dist.values())
        
        entropy = 0
        for count in freq_dist.values():
            p = count / total_count
            entropy += -p * math.log(p)
            
        return entropy

    def self_bleu(self, n_gram=4, sample_size=500):
        """
        计算 Self-BLEU。
        为了避免计算爆炸 (O(N^2))，如果数据量大，我们进行采样计算。
        :param sample_size: 随机抽样的样本数量，设为 None 则计算全量。
        """
        if self.num_docs < 2: return 0.0
        
        # 如果文本太多，进行采样以加快速度
        if sample_size and self.num_docs > sample_size:
            sampled_indices = random.sample(range(self.num_docs), sample_size)
            pool = [self.tokenized_texts[i] for i in sampled_indices]
        else:
            pool = self.tokenized_texts

        bleu_scores = []
        smooth = SmoothingFunction().method1
        
        for i, hypothesis in enumerate(pool):
            # 将除自己以外的所有句子作为 references
            references = pool[:i] + pool[i+1:]
            if not references: continue
            
            # 计算当前句子与其余句子的 BLEU score
            score = sentence_bleu(references, hypothesis, weights=(1./n_gram,)*n_gram, smoothing_function=smooth)
            bleu_scores.append(score)
            
        return np.mean(bleu_scores)

    def semantic_similarity(self):
        """
        基于 TF-IDF 的平均余弦相似度。
        Self-BLEU 衡量字面重合，这个指标衡量语义重复度。
        值越高，多样性越差（说明大家都在说车轱辘话）。
        """
        if self.num_docs < 2: return 0.0
        
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = tfidf.fit_transform(self.raw_texts)
        except ValueError:
            # 处理空词汇表情况
            return 0.0
            
        # 计算两两相似度矩阵
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        
        # 获取上三角矩阵的平均值（不包含对角线的自身相似度）
        upper_tri_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
        mean_similarity = np.mean(cosine_sim_matrix[upper_tri_indices])
        
        return mean_similarity

def process_file(file_path):
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: File not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return

    # 提取并拼接 relationship 和 survival 文本
    combined_texts = []

    for patient_id, analyses in data.items():
        for analysis in analyses:
            # 获取两个字段的文本，如果不存在则为空字符串
            rel_text = analysis.get('relationship', '')
            surv_text = analysis.get('survival', '')
            
            # 拼接文本，中间加一个空格
            # 只有当至少有一个字段非空时才添加
            full_text = f"{rel_text} {surv_text}".strip()
            if full_text:
                combined_texts.append(full_text)

    print(f"\nTotal Combined samples: {len(combined_texts)}")
    print("-" * 50)

    # 定义要评估的类别 - 这里只计算拼接后的结果
    categories = {
        "Combined Relationship & Survival": combined_texts
    }

    results = {}

    for name, texts in categories.items():
        print(f"\nCalculating metrics for: {name}...")
        evaluator = DiversityEvaluator(texts)
        
        metrics = {
            "Vocabulary Size (higher is better)": evaluator.vocab_size(),
            "Distinct-1 (higher is better)": evaluator.distinct_n(1),
            "Distinct-2 (higher is better)": evaluator.distinct_n(2),
            "Entropy-4 (higher is better)": evaluator.entropy_n(4),
            "Self-BLEU-4 (LOWER is better)": evaluator.self_bleu(n_gram=4, sample_size=200),
            "Cosine Similarity (LOWER is better)": evaluator.semantic_similarity()
        }
        
        results[name] = metrics
        
        for k, v in metrics.items():
            print(f"  {k:<35}: {v:.4f}")

if __name__ == "__main__":
    # 你指定的文件路径
    FILE_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/medical_analysis_deepseek.json"
    process_file(FILE_PATH)