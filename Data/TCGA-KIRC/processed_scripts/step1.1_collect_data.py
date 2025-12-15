import pandas as pd
import numpy as np
import os
import string

# =============================================================================
# 1. 配置与常量定义 (Configuration & Constants)
# =============================================================================

# ID Keys
KEY_CASE_ID = 'cases.case_id'
KEY_SUBMITTER_ID = 'cases.submitter_id'

# ---------------------------------------------------------
# 诊断信息列定义 (Diagnostic Columns)
# ---------------------------------------------------------
diagnostic_columns_raw = [
    'cases.case_id#病例ID',
    'cases.submitter_id#提交者ID',
    'cases.disease_type#疾病类型',
    'cases.primary_site#原发部位',
    'demographic.age_at_index#入组时年龄',
    'demographic.age_is_obfuscated#年龄是否脱敏',
    'demographic.country_of_birth#出生国家',
    'demographic.country_of_residence_at_enrollment#入组时居住国家',
    'demographic.education_level#教育程度',
    'demographic.ethnicity#民族',
    'demographic.gender#性别',
    'demographic.marital_status#婚姻状况',
    'demographic.population_group#人群分组',
    'demographic.race#种族',
    'exposures.alcohol_days_per_week#每周饮酒天数',
    'exposures.alcohol_drinks_per_day#每日饮酒量',
    'exposures.alcohol_frequency#饮酒频率',
    'exposures.alcohol_history#饮酒史',
    'exposures.alcohol_intensity#饮酒强度',
    'exposures.alcohol_type#饮酒类型',
    'exposures.asbestos_exposure#石棉暴露',
    'exposures.asbestos_exposure_type#石棉暴露类型',
    'exposures.chemical_exposure_type#化学暴露类型',
    'exposures.cigarettes_per_day#每日吸烟量',
    'exposures.coal_dust_exposure#煤尘暴露',
    'exposures.environmental_tobacco_smoke_exposure#环境烟草烟雾暴露',
    'exposures.exposure_duration#暴露持续时间',
    'exposures.exposure_duration_hrs_per_day#每日暴露时长（小时）',
    'exposures.exposure_source#暴露来源',
    'exposures.exposure_type#暴露类型',
    'exposures.occupation_type#职业类型',
    'exposures.parent_with_radiation_exposure#父母辐射暴露史',
    'exposures.radon_exposure#氡暴露',
    'exposures.respirable_crystalline_silica_exposure#可吸入结晶二氧化硅暴露',
    'exposures.secondhand_smoke_as_child#儿童期二手烟暴露',
    'exposures.smoking_frequency#吸烟频率',
    'exposures.tobacco_smoking_status#吸烟状态',
    'exposures.type_of_smoke_exposure#烟雾暴露类型',
    'exposures.type_of_tobacco_used#烟草类型',
    'other_clinical_attributes.aids_risk_factors#艾滋病风险因素',
    'other_clinical_attributes.bmi#身体质量指数（BMI）',
    'other_clinical_attributes.body_surface_area#体表面积',
    'other_clinical_attributes.cd4_count#CD4细胞计数',
    'other_clinical_attributes.cdc_hiv_risk_factors#CDC HIV风险因素',
    'other_clinical_attributes.comorbidities#合并症',
    'other_clinical_attributes.comorbidity_method_of_diagnosis#合并症诊断方法',
    'other_clinical_attributes.diabetes_treatment_type#糖尿病治疗类型（基线状态）',
    'other_clinical_attributes.dlco_ref_predictive_percent#DLCO预测百分比（肺功能）',
    'other_clinical_attributes.exercise_frequency_weekly#每周锻炼频率',
    'other_clinical_attributes.eye_color#虹膜颜色',
    'other_clinical_attributes.hiv_viral_load#HIV病毒载量',
    'other_clinical_attributes.menopause_status#绝经状态',
    'other_clinical_attributes.myasthenia_gravis_classification#重症肌无力分类',
    'other_clinical_attributes.nadir_cd4_count#CD4最低计数',
    'other_clinical_attributes.nononcologic_therapeutic_agents#非肿瘤治疗药物（基线）',
    'other_clinical_attributes.number_of_pregnancies#妊娠次数',
    'other_clinical_attributes.oxygen_use_indicator#吸氧指征（基线）',
    'other_clinical_attributes.oxygen_use_type#吸氧类型（基线）',
    'other_clinical_attributes.pregnancy_count#妊娠次数',
    'other_clinical_attributes.pregnancy_outcome#妊娠结局',
    'other_clinical_attributes.premature_at_birth#出生时早产',
    'other_clinical_attributes.risk_factor_method_of_diagnosis#风险因素诊断方法',
    'other_clinical_attributes.risk_factors#风险因素',
    'other_clinical_attributes.undescended_testis_history#隐睾病史',
    'other_clinical_attributes.undescended_testis_history_laterality#隐睾侧别',
    'other_clinical_attributes.viral_hepatitis_serologies#病毒性肝炎血清学',
    'other_clinical_attributes.viral_hepatitis_serology_tests#肝炎血清学检测',
    'other_clinical_attributes.weeks_gestation_at_birth#出生孕周',
    'other_clinical_attributes.weight#体重',
]

# ---------------------------------------------------------
# 治疗信息列定义 (Treatment Columns)
# ---------------------------------------------------------
treatment_columns_raw = [
    'other_clinical_attributes.haart_treatment_indicator#高效抗逆转录病毒治疗指征',
    'other_clinical_attributes.hepatitis_sustained_virological_response#肝炎持续病毒学应答',
    'other_clinical_attributes.hormonal_contraceptive_type#激素避孕类型',
    'other_clinical_attributes.hormonal_contraceptive_use#激素避孕使用',
    'other_clinical_attributes.hormonal_replacement_therapy_status#激素替代治疗状态',
    'other_clinical_attributes.hormone_replacement_therapy_type#激素替代治疗类型',
    'other_clinical_attributes.hysterectomy_margins_involved#子宫切除术切缘受累',
    'other_clinical_attributes.hysterectomy_type#子宫切除术类型',
    'other_clinical_attributes.immunosuppressive_treatment_type#免疫抑制治疗类型',
    'other_clinical_attributes.reflux_treatment_type#反流治疗类型',
    'other_clinical_attributes.risk_factor_treatment#风险因素治疗',
    'other_clinical_attributes.treatment_frequency#治疗频率',
    'other_clinical_attributes.undescended_testis_corrected#隐睾矫正术',
    'other_clinical_attributes.undescended_testis_corrected_age_range#隐睾矫正年龄范围',
    'other_clinical_attributes.undescended_testis_corrected_laterality#隐睾矫正侧别',
    'other_clinical_attributes.undescended_testis_corrected_method#隐睾矫正方法',
    'treatments.chemo_concurrent_to_radiation#同步放化疗',
    'treatments.clinical_trial_indicator#临床试验标识',
    'treatments.course_number#治疗疗程编号',
    'treatments.drug_category#药物类别',
    'treatments.embolic_agent#栓塞剂',
    'treatments.number_of_cycles#治疗周期数',
    'treatments.number_of_fractions#放疗分割次数',
    'treatments.prescribed_dose#处方剂量',
    'treatments.pretreatment#预处理方案',
    'treatments.protocol_identifier#治疗方案标识',
    'treatments.radiosensitizing_agent#放射增敏剂',
    'treatments.regimen_or_line_of_therapy#治疗方案/治疗线数',
    'treatments.route_of_administration#给药途径',
    'treatments.therapeutic_level_achieved#治疗水平达成',
    'treatments.therapeutic_levels_achieved#治疗水平达成（复数）',
    'treatments.therapeutic_target_level#治疗靶目标水平',
    'treatments.treatment_anatomic_site#治疗解剖部位',
    'treatments.treatment_anatomic_sites#治疗解剖部位（多部位）',
    'treatments.treatment_arm#治疗组别',
    'treatments.treatment_dose#治疗剂量',
    'treatments.treatment_dose_max#最大治疗剂量',
    'treatments.treatment_duration#治疗持续时间',
    'treatments.treatment_frequency#治疗频率',
    'treatments.treatment_or_therapy#治疗或疗法类型',
    'treatments.treatment_intent_type',
    "treatments.treatment_type",
    'treatments.therapeutic_agents#治疗药物',
]

# ---------------------------------------------------------
# 文本类治疗信息 (Text Treatment Columns)  
# ---------------------------------------------------------
treatment_text_columns_raw = [
    'treatments.treatment_or_therapy#治疗或疗法类型',
    'treatments.treatment_intent_type',
    "treatments.treatment_type",
    'treatments.therapeutic_agents#治疗药物',
]

# ---------------------------------------------------------
# 病理信息列定义 (Pathology Columns)
# ---------------------------------------------------------
pathology_columns_raw = [
    'diagnoses.adrenal_hormone#肾上腺激素水平',
    'diagnoses.age_at_diagnosis#诊断时年龄',
    'diagnoses.ajcc_clinical_m#AJCC临床M分期（远处转移）',
    'diagnoses.ajcc_clinical_n#AJCC临床N分期（淋巴结转移）',
    'diagnoses.ajcc_clinical_stage#AJCC临床分期',
    'diagnoses.ajcc_clinical_t#AJCC临床T分期（原发肿瘤）',
    
    'diagnoses.ajcc_pathologic_m#AJCC病理M分期',
    'diagnoses.ajcc_pathologic_n#AJCC病理N分期',
    'diagnoses.ajcc_pathologic_stage#AJCC病理分期',
    'diagnoses.ajcc_pathologic_t#AJCC病理T分期',

    'diagnoses.ajcc_serum_tumor_markers#AJCC血清肿瘤标志物',
    'diagnoses.ajcc_staging_system_edition#AJCC分期系统版本',
    'diagnoses.ann_arbor_b_symptoms#Ann Arbor B症状（淋巴瘤）',
    'diagnoses.ann_arbor_b_symptoms_described#B症状详细描述',
    'diagnoses.ann_arbor_clinical_stage#Ann Arbor临床分期',
    'diagnoses.ann_arbor_extranodal_involvement#结外侵犯',

    'diagnoses.ann_arbor_pathologic_stage#Ann Arbor病理分期',
    
    'diagnoses.burkitt_lymphoma_clinical_variant#伯基特淋巴瘤临床亚型',
    'diagnoses.calgb_risk_group#CALGB风险分组',
    'diagnoses.cancer_detection_method#癌症检测方法',
    'diagnoses.child_pugh_classification#Child-Pugh肝功能分级',
    'diagnoses.clark_level#Clark浸润深度分级（黑色素瘤）',
    'diagnoses.classification_of_tumor#肿瘤分类',
    'diagnoses.cog_liver_stage#COG肝母细胞瘤分期',
    'diagnoses.cog_neuroblastoma_risk_group#COG神经母细胞瘤风险组',
    'diagnoses.cog_renal_stage#COG肾母细胞瘤分期',
    'diagnoses.cog_rhabdomyosarcoma_risk_group#COG横纹肌肉瘤风险组',
    'diagnoses.contiguous_organ_invaded#邻近器官侵犯',
    'diagnoses.diagnosis_is_primary_disease#是否原发疾病',
    'diagnoses.double_expressor_lymphoma#双表达淋巴瘤标志',
    'diagnoses.double_hit_lymphoma#双打击淋巴瘤标志',
    'diagnoses.eln_risk_classification#ELN风险分层（白血病）',
    'diagnoses.enneking_msts_grade#Enneking骨肿瘤分级',
    'diagnoses.enneking_msts_metastasis#Enneking转移状态',
    'diagnoses.enneking_msts_stage#Enneking骨肿瘤分期',
    'diagnoses.enneking_msts_tumor_site#Enneking肿瘤部位',
    'diagnoses.ensat_clinical_m#ENSAT临床M分期（肾上腺肿瘤）',
    
    'diagnoses.ensat_pathologic_n#ENSAT病理N分期',
    'diagnoses.ensat_pathologic_stage#ENSAT病理分期',
    'diagnoses.ensat_pathologic_t#ENSAT病理T分期',

    'diagnoses.esophageal_columnar_dysplasia_degree#食管柱状上皮异型增生程度',
    'diagnoses.esophageal_columnar_metaplasia_present#食管柱状上皮化生存在',
    'diagnoses.fab_morphology_code#FAB形态学编码（白血病）',
    'diagnoses.figo_stage#FIGO妇科肿瘤分期',
    'diagnoses.first_symptom_longest_duration#最长症状持续时间',
    'diagnoses.first_symptom_prior_to_diagnosis#诊断前首发症状',
    'diagnoses.gastric_esophageal_junction_involvement#胃食管结合部侵犯',
    'diagnoses.gleason_grade_group#Gleason分级组（前列腺癌）',
    'diagnoses.gleason_grade_tertiary#Gleason三级评分',
    'diagnoses.gleason_patterns_percent#Gleason模式百分比',
    'diagnoses.gleason_score#Gleason评分',
    'diagnoses.goblet_cells_columnar_mucosa_present#杯状细胞柱状黏膜存在',
    'diagnoses.icd_10_code#ICD-10疾病编码',
    'diagnoses.igcccg_stage#IGCCCG分期（生殖细胞肿瘤）',
    'diagnoses.inpc_grade#INPC组织学分级（肾母细胞瘤）',
    'diagnoses.inpc_histologic_group#INPC组织学分组',
    'diagnoses.inrg_stage#INRG分期（神经母细胞瘤）',
    'diagnoses.inss_stage#INSS分期（神经母细胞瘤）',
    'diagnoses.international_prognostic_index#国际预后指数（IPI）',
    'diagnoses.irs_group#IRS分组（横纹肌肉瘤）',
    'diagnoses.irs_stage#IRS分期',
    'diagnoses.ishak_fibrosis_score#Ishak肝纤维化评分',
    'diagnoses.iss_stage#ISS分期（多发性骨髓瘤）',
    'diagnoses.laterality#肿瘤侧别',

    'diagnoses.margin_distance#切缘距离（诊断时评估）',
    'diagnoses.margins_involved_site#受累切缘部位（诊断时）',

    'diagnoses.masaoka_stage#Masaoka胸腺瘤分期',
    'diagnoses.max_tumor_bulk_site#最大肿瘤负荷部位',
    'diagnoses.medulloblastoma_molecular_classification#髓母细胞瘤分子分型',
    'diagnoses.melanoma_known_primary#黑色素瘤已知原发灶',
    'diagnoses.metastasis_at_diagnosis#诊断时转移状态',
    'diagnoses.metastasis_at_diagnosis_site#转移部位（诊断时）',
    'diagnoses.method_of_diagnosis#诊断方法',
    'diagnoses.micropapillary_features#微乳头状特征',
    'diagnoses.mitosis_karyorrhexis_index#核分裂-核碎裂指数',
    'diagnoses.mitotic_count#核分裂计数',
    'diagnoses.morphology#形态学类型',
    'diagnoses.ovarian_specimen_status#卵巢标本状态',
    'diagnoses.ovarian_surface_involvement#卵巢表面受累',
    'diagnoses.papillary_renal_cell_type#乳头状肾细胞癌亚型',
    'diagnoses.pediatric_kidney_staging#儿童肾肿瘤分期',
    'diagnoses.peritoneal_fluid_cytological_status#腹腔积液细胞学状态',
    'diagnoses.pregnant_at_diagnosis#诊断时妊娠状态',
    'diagnoses.primary_diagnosis#原发诊断',
    'diagnoses.primary_disease#原发疾病',
    'diagnoses.primary_gleason_grade#Gleason主要分级',
    'diagnoses.prior_malignancy#既往恶性肿瘤史',
    'diagnoses.prior_treatment#既往治疗史（诊断前）',
    'diagnoses.satellite_nodule_present#卫星结节存在',
    'diagnoses.secondary_gleason_grade#Gleason次要分级',
    'diagnoses.site_of_resection_or_biopsy#切除/活检部位',
    'diagnoses.sites_of_involvement#受累部位',
    'diagnoses.sites_of_involvement_count#受累部位数量',
    'diagnoses.supratentorial_localization#幕上定位（脑肿瘤）',
    'diagnoses.synchronous_malignancy#同步原发肿瘤',
    'diagnoses.tissue_or_organ_of_origin#起源组织/器官',

    'diagnoses.tumor_burden#肿瘤负荷',
    'diagnoses.tumor_confined_to_organ_of_origin#肿瘤局限于起源器官',
    'diagnoses.tumor_depth#肿瘤浸润深度',
    'diagnoses.tumor_focality#肿瘤灶性（单发/多发）',
    'diagnoses.tumor_grade#肿瘤分级',
    'diagnoses.tumor_grade_category#肿瘤分级类别',
    'diagnoses.tumor_of_origin#起源肿瘤',
    'diagnoses.tumor_regression_grade#肿瘤退缩分级',

    'diagnoses.uicc_clinical_m#UICC临床M分期',
    'diagnoses.uicc_clinical_n#UICC临床N分期',
    'diagnoses.uicc_clinical_stage#UICC临床分期',
    'diagnoses.uicc_clinical_t#UICC临床T分期',
    
    'diagnoses.uicc_pathologic_m#UICC病理M分期',
    'diagnoses.uicc_pathologic_n#UICC病理N分期',
    'diagnoses.uicc_pathologic_stage#UICC病理分期',
    'diagnoses.uicc_pathologic_t#UICC病理T分期',

    'diagnoses.uicc_staging_system_edition#UICC分期系统版本',
    'diagnoses.ulceration_indicator#溃疡指标（黑色素瘤）',
    'diagnoses.weiss_assessment_findings#Weiss评估发现（肾上腺皮质癌）',
    'diagnoses.weiss_assessment_score#Weiss评分',
    'diagnoses.who_cns_grade#WHO中枢神经系统肿瘤分级',
    'diagnoses.who_nte_grade#WHO非睾丸生殖细胞瘤分级',
    'diagnoses.wilms_tumor_histologic_subtype#肾母细胞瘤组织学亚型',

    'pathology_details.additional_pathology_findings',      # 附加病理发现（如特殊细胞类型）
    'pathology_details.anaplasia_present',                  # 间变存在（细胞异型性）
    'pathology_details.anaplasia_present_type',             # 间变类型
    'pathology_details.columnar_mucosa_present',            # 柱状黏膜存在（如Barrett食管）
    'pathology_details.consistent_pathology_review',        # 病理复核一致性（报告流程）
    'pathology_details.dysplasia_degree',                   # 异型增生程度（癌前病变）
    'pathology_details.dysplasia_type',                     # 异型增生类型
    'pathology_details.epithelioid_cell_percent',           # 上皮样细胞百分比
    'pathology_details.epithelioid_cell_percent_range',     # 上皮样细胞百分比范围
    'pathology_details.histologic_progression_type',        # 组织学进展类型
    'pathology_details.intratubular_germ_cell_neoplasia_present',# 管内生殖细胞瘤（睾丸活检特有）
    'pathology_details.lymphatic_invasion_present',         # 淋巴管侵犯（若活检包含血管/淋巴管）
    'pathology_details.metaplasia_present',                 # 化生存在（如肠上皮化生）
    'pathology_details.morphologic_architectural_pattern',  # 组织学结构模式
    'pathology_details.necrosis_percent',                   # 坏死百分比（局部区域）
    'pathology_details.necrosis_present',                   # 坏死存在
    'pathology_details.number_proliferating_cells',         # 增殖细胞数量（如Ki-67染色）
    'pathology_details.percent_tumor_nuclei',               # 肿瘤细胞核百分比（活检样本内）
    'pathology_details.perineural_invasion_present',        # 神经周围侵犯（若活检包含神经）
    'pathology_details.prcc_type',                          # 乳头状肾细胞癌分型（肾活检特有）
    'pathology_details.rhabdoid_percent',                   # 横纹肌样细胞百分比
    'pathology_details.rhabdoid_present',                   # 横纹肌样特征存在
    'pathology_details.sarcomatoid_percent',                # 肉瘤样细胞百分比
    'pathology_details.sarcomatoid_present',                # 肉瘤样特征存在
    'pathology_details.spindle_cell_percent',               # 梭形细胞百分比
    'pathology_details.spindle_cell_percent_range',         # 梭形细胞百分比范围
    'pathology_details.tumor_infiltrating_lymphocytes',     # 肿瘤浸润淋巴细胞
    'pathology_details.tumor_infiltrating_macrophages',     # 肿瘤浸润巨噬细胞
    'pathology_details.vascular_invasion_present',          # 血管侵犯（若活检包含血管）
    'pathology_details.vascular_invasion_type',             # 血管侵犯类型
    'pathology_details.zone_of_origin_prostate',            # 前列腺起源区（前列腺活检特有）
    
    'pathology_details.prostatic_chips_positive_count',     # 阳性穿刺条数
    'pathology_details.prostatic_chips_total_count',        # 总穿刺条数
    'pathology_details.prostatic_involvement_percent'       # 肿瘤累及百分比
]

# 清理列名，去掉#和中文注释，并去除空白
diagnostic_columns = [col.split('#')[0].strip() for col in diagnostic_columns_raw]
treatment_columns = [col.split('#')[0].strip() for col in treatment_columns_raw]
treatment_text_columns = [col.split('#')[0].strip() for col in treatment_text_columns_raw]
pathology_columns = [col.split('#')[0].strip() for col in pathology_columns_raw]

# 确保列表示唯一
diagnostic_columns = list(set(diagnostic_columns))
treatment_columns = list(set(treatment_columns))
treatment_text_columns = list(set(treatment_text_columns))
pathology_columns = list(set(pathology_columns))

print(f"Loaded {len(diagnostic_columns)} diagnostic columns.")
print(f"Loaded {len(treatment_columns)} treatment columns.")
print(f"Loaded {len(treatment_text_columns)} treatment text columns.")
print(f"Loaded {len(pathology_columns)} pathology columns.")


# =============================================================================
# 2. 功能函数定义 (Helper Functions)
# =============================================================================

def aggregate_patient_data(df, key_column, columns_to_process):
    """
    按键列聚合 DataFrame。
    对于数值列：计算均值和方差。
    对于非数值列：获取唯一非空值，排序并用 '+' 连接。
    """
    print(f"Starting aggregation on key: {key_column}")
    
    if key_column not in df.columns:
        print(f"Error: Key column '{key_column}' not in DataFrame.")
        return None
        
    # 创建包含唯一键的基础 DataFrame
    unique_keys = df[[key_column]].drop_duplicates().reset_index(drop=True)
    
    processed_count = 0
    for col in columns_to_process:
        if col == key_column:
            continue
            
        if col not in df.columns:
            # 很多列可能在当前项目中不存在，静默跳过或仅调试时输出
            # print(f"Skipping missing column: {col}")
            continue

        processed_count += 1

        # 1. 尝试转换为数值
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        # 2. 检查是否为数值列 (如果全是非空且是数字)
        # 注意：如果一列全是NaN，to_numeric后也是NaN，这里需要判断是否有有效数值
        if not numeric_series.dropna().empty and numeric_series.notnull().sum() > 0:
            # 简单判断：如果转换后，原本非空的值大部分还在，说明是数值列
            # 这里简化逻辑：只要能转成数值且不全是NaN，就算数值处理
             print(f"   -> Processing '{col}' as NUMERIC (mean/variance)...")
             temp_df = pd.DataFrame({
                 key_column: df[key_column],
                 col: numeric_series
             })
             
             stats = temp_df.groupby(key_column)[col].agg(['mean', 'var']).reset_index()
             stats.rename(columns={
                 'mean': f'{col}_mean',
                 'var': f'{col}_variance'
             }, inplace=True)
             
             unique_keys = pd.merge(unique_keys, stats, on=key_column, how='left')
            
        else:
            # 3. 处理为非数值列 (字符串拼接)
            print(f"   -> Processing '{col}' as NON-NUMERIC (joining unique values)...")

            def join_unique_strings(series):
                non_null_series = series.dropna()
                if non_null_series.empty:
                    return pd.NA
                
                filtered_strings = set()
                for s in non_null_series.astype(str):
                    stripped_s = s.strip()
                    if stripped_s and stripped_s.lower() != 'nan' and stripped_s.lower() != 'none':
                         # 简单的去除仅包含标点符号的逻辑
                        if not all(char in string.punctuation for char in stripped_s):
                            filtered_strings.add(stripped_s)
                
                if not filtered_strings:
                    return pd.NA
                
                return '+'.join(sorted(list(filtered_strings)))

            relevant_data = df[[key_column, col]]
            aggregated_col = relevant_data.groupby(key_column)[col].apply(join_unique_strings).reset_index()
            unique_keys = pd.merge(unique_keys, aggregated_col, on=key_column, how='left')

    print(f"Aggregation complete. Processed {processed_count} columns found in source.")
    return unique_keys


def process_mixed_types_and_encode(df: pd.DataFrame, keys_to_preserve: list = None, verbose: bool = True) -> pd.DataFrame:
    """
    处理混合数据类型并进行编码。
    1. 尝试将 object 列转换为数值。
    2. 无法转换的视为分类列，编码为 1-based 整数。
    3. NaN 值替换为 -1。
    4. 跳过 keys_to_preserve 中的列。
    """
    df_processed = df.copy()
    if keys_to_preserve is None:
        keys_to_preserve = []
        
    if verbose:
        print(f"\n--- 🔄 开始处理数据类型 (共 {df_processed.shape[1]} 列) ---")
    
    encoded_cols = []
    converted_cols = []
    key_cols_skipped = []

    for col in df_processed.columns:
        if col in keys_to_preserve:
            key_cols_skipped.append(col)
            continue

        if df_processed[col].dtype == 'object':
            # 尝试转为数值
            col_numeric = pd.to_numeric(df_processed[col], errors='coerce')
            
            # 检查是否因为无法转换变成了NaN（原先不是NaN，现在是NaN的比例）
            # 如果原列有值，但转换后全是NaN，说明是纯字符串
            if col_numeric.notna().sum() > 0:
                # 混合情况：包含一部分数字和一部分非数字，或者全是数字存成的字符串
                # 这里我们简单策略：如果能转成数字的比例较高，则保留数值，无法转换的填-1
                # 但更安全的做法：如果包含无法转换的字符串，作为Categorical编码
                
                # 检查是否有“真正的”非数字字符串
                non_numeric_mask = pd.to_numeric(df_processed[col], errors='coerce').isna() & df_processed[col].notna()
                
                if not non_numeric_mask.any():
                    # 情况A: 纯数值型 (存储为字符串)
                    df_processed[col] = col_numeric.fillna(-1)
                    converted_cols.append(col)
                else:
                    # 情况B: 混合型或类别型 -> 编码
                    codes, uniques = pd.factorize(df_processed[col], sort=True)
                    encoded_series = pd.Series(codes, index=df_processed.index)
                    # factorize中 -1 代表缺失，我们想要缺失也是 -1，其他从 1 开始
                    # factorize 的缺失默认是 -1. 
                    # 我们希望有效值从1开始，缺失值保持-1
                    # 原codes: 缺失=-1, A=0, B=1
                    # 目标: 缺失=-1, A=1, B=2
                    
                    # 将非-1的值加1
                    encoded_series = encoded_series.where(encoded_series == -1, encoded_series + 1)
                    
                    df_processed[col] = encoded_series
                    encoded_cols.append(col)
            else:
                 # 纯字符列
                codes, uniques = pd.factorize(df_processed[col], sort=True)
                encoded_series = pd.Series(codes, index=df_processed.index)
                encoded_series = encoded_series.where(encoded_series == -1, encoded_series + 1)
                df_processed[col] = encoded_series
                encoded_cols.append(col)
        else:
            # 已经是数值型，只需填充
            pass
                
    # 处理原本就是数值型但包含 NaN 的列
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col in keys_to_preserve:
            continue
            
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(-1)
                
    if verbose:
        if key_cols_skipped:
            print(f"  > 保持 [Key] 列为原始格式: {len(key_cols_skipped)} 列")
        if converted_cols:
            print(f"  > 成功转换为 [数值型] (NaN -> -1): {len(converted_cols)} 列")
        if encoded_cols:
            print(f"  > 成功编码为 [分类型] (1-based, NaN -> -1): {len(encoded_cols)} 列")
    
    return df_processed


def filter_unsupervised_features(
    df: pd.DataFrame, 
    missing_threshold: float = 0.9, 
    cardinality_threshold: float = 0.95, 
    variance_threshold: float = 0.0, 
    verbose: bool = True,
    keep_cols: list = None
) -> pd.DataFrame:
    """
    无监督特征过滤：高缺失率、高基数、低方差。
    将 -1 视为空值。
    """
    if keep_cols is None:
        keep_cols = []
        
    original_cols = set(df.columns)
    total_rows = len(df)
    df_filtered = df.copy()
    dropped_cols_log = []
    
    if total_rows == 0:
        return df_filtered

    # 1. 过滤高缺失率 (-1)
    missing_ratios = (df_filtered == -1).sum() / total_rows
    cols_to_drop_missing = [c for c in missing_ratios[missing_ratios > missing_threshold].index if c not in keep_cols]
    
    for col in cols_to_drop_missing:
        dropped_cols_log.append((col, f"缺失率 > {missing_threshold}", f"{missing_ratios[col]:.2%}"))
    
    df_filtered = df_filtered.drop(columns=cols_to_drop_missing)

    # 2. 过滤高基数 (ID列，忽略 -1)
    unique_ratios = df_filtered.apply(lambda col: col[col != -1].nunique()) / total_rows
    cols_to_drop_cardinality = [c for c in unique_ratios[unique_ratios > cardinality_threshold].index if c not in keep_cols]
    
    for col in cols_to_drop_cardinality:
        dropped_cols_log.append((col, f"唯一值比例 > {cardinality_threshold}", f"{unique_ratios[col]:.2%}"))
        
    df_filtered = df_filtered.drop(columns=cols_to_drop_cardinality)

    # 3. 过滤低方差 (忽略 -1)
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    df_numeric = df_filtered[numeric_cols]
    
    if not df_numeric.empty:
        try:
            df_numeric_temp = df_numeric.replace(-1, np.nan)
            variances = df_numeric_temp.var(skipna=True, ddof=0)
            
            cols_to_drop_numeric = []
            candidates = variances[(variances <= variance_threshold) | (variances.isnull())].index
            
            for col in candidates:
                if col not in keep_cols:
                    cols_to_drop_numeric.append(col)
                    val_str = f"{variances.get(col):.4f}" if pd.notnull(variances.get(col)) else "NaN"
                    dropped_cols_log.append((col, f"方差 <= {variance_threshold}", val_str))
            
            df_filtered = df_filtered.drop(columns=cols_to_drop_numeric)
        except ValueError as e:
            if verbose: print(f"警告: 方差过滤失败. {e}")

    # 4. 过滤非数值常量 (忽略 -1)
    non_numeric_cols = df_filtered.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        nunique = df_filtered[non_numeric_cols].apply(lambda col: col[col != -1].nunique(dropna=True))
        cols_to_drop = [c for c in nunique[nunique <= 1].index if c not in keep_cols]
        
        for col in cols_to_drop:
            dropped_cols_log.append((col, "常量 (非数值型)", f"{nunique[col]} 个唯一值"))
            
        df_filtered = df_filtered.drop(columns=cols_to_drop)

    if verbose:
        print("\n--- 🤖 无监督特征过滤报告 ---")
        if dropped_cols_log:
            print(f"🗑️ 已删除 {len(dropped_cols_log)} 列")
            # 可选：打印前几个删除的列
            # for log in dropped_cols_log[:5]:
            #    print(f"   - {log}")
        print(f"原始列数: {len(original_cols)} -> 最终保留: {len(df_filtered.columns)}")
        print("---------------------------------------")

    return df_filtered


# =============================================================================
# 3. 主执行流程 (Main Execution)
# =============================================================================

def main():
    # --- 路径设置 ---
    # !! 重要: 请更新为您的实际路径 !!
    base_path = '/home/Zhengzx/MedAlignFusion/Data/TCGA-KIRC/source'
    save_path = '/home/Zhengzx/MedAlignFusion/Data/TCGA-KIRC/processed'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created save directory: {save_path}")
    
    # 原始文件路径
    biospecimen_dir = os.path.join(base_path, 'biospecimen.project-tcga-kirc.2025-12-11')
    biospecimen_files = [
        os.path.join(biospecimen_dir, f) for f in ['aliquot.tsv', 'analyte.tsv', 'portion.tsv', 'sample.tsv', 'slide.tsv']
    ]
    clinical_dir = os.path.join(base_path, 'clinical.project-tcga-kirc.2025-12-11')
    clinical_files = [
        os.path.join(clinical_dir, f) for f in ['clinical.tsv', 'exposure.tsv', 'family_history.tsv', 'follow_up.tsv', 'pathology_detail.tsv']
    ]
    all_files = biospecimen_files + clinical_files

    # --- 阶段 1: 构建 ID 映射 ---
    print("\n[Step 1/5] Building ID map...")
    case_to_submitter = {}
    submitter_to_case = {}
    
    for file_path in all_files:
        if not os.path.exists(file_path):
            continue
        try:
            df_ids = pd.read_csv(file_path, sep='\t', header=0, low_memory=False, 
                               usecols=lambda c: c in [KEY_CASE_ID, KEY_SUBMITTER_ID])
            if KEY_CASE_ID in df_ids.columns and KEY_SUBMITTER_ID in df_ids.columns:
                id_pairs = df_ids[[KEY_CASE_ID, KEY_SUBMITTER_ID]].dropna().drop_duplicates().values
                for case, submitter in id_pairs:
                    case_to_submitter[case] = submitter
                    submitter_to_case[submitter] = case
        except Exception:
            pass

    print(f"ID map built: {len(case_to_submitter)} entries.")

    # --- 阶段 2: 加载与合并原始数据 ---
    print("\n[Step 2/5] Loading and merging files...")
    dfs_to_merge = []
    all_loaded_columns = set([KEY_CASE_ID, KEY_SUBMITTER_ID])

    for file_path in all_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path, sep='\t', header=0, low_memory=False)
            
            # 补全 ID
            if KEY_CASE_ID in df.columns and KEY_SUBMITTER_ID not in df.columns:
                df[KEY_SUBMITTER_ID] = df[KEY_CASE_ID].map(case_to_submitter)
            elif KEY_CASE_ID not in df.columns and KEY_SUBMITTER_ID in df.columns:
                df[KEY_CASE_ID] = df[KEY_SUBMITTER_ID].map(submitter_to_case)

            if KEY_CASE_ID in df.columns and KEY_SUBMITTER_ID in df.columns:
                # 仅保留新列以节省内存
                new_cols = [c for c in df.columns if c not in all_loaded_columns]
                cols_to_keep = [KEY_CASE_ID, KEY_SUBMITTER_ID] + new_cols
                cols_to_keep_existing = [c for c in cols_to_keep if c in df.columns]
                
                df_subset = df[cols_to_keep_existing]
                all_loaded_columns.update(new_cols)
                dfs_to_merge.append(df_subset)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not dfs_to_merge:
        print("No data loaded. Exiting.")
        return

    merged_df = pd.concat(dfs_to_merge, ignore_index=True, sort=False)
    merged_df.drop_duplicates(inplace=True)
    print(f"Merged DataFrame shape: {merged_df.shape}")

    # --- 阶段 3: 数据聚合 (Aggregation) ---
    print("\n[Step 3/5] Aggregating Data...")
    
    cols_to_agg_diagnostic = [c for c in diagnostic_columns if c != KEY_CASE_ID]
    cols_to_agg_treatment = [c for c in treatment_columns if c != KEY_CASE_ID]
    cols_to_agg_pathology = [c for c in pathology_columns if c != KEY_CASE_ID]
    cols_to_agg_treatment_text = [c for c in treatment_text_columns if c != KEY_CASE_ID] # 新增 text 处理

    # 聚合得到初步 DataFrame
    print("\n>>> Aggregating Clinical/Diagnostic Data...")
    diagnostic_agg_df = aggregate_patient_data(merged_df, KEY_CASE_ID, cols_to_agg_diagnostic)
    
    print("\n>>> Aggregating Treatment Data...")
    treatment_agg_df = aggregate_patient_data(merged_df, KEY_CASE_ID, cols_to_agg_treatment)
    
    print("\n>>> Aggregating Pathology Data...")
    pathology_agg_df = aggregate_patient_data(merged_df, KEY_CASE_ID, cols_to_agg_pathology)

    print("\n>>> Aggregating Treatment Text Data...") # 新增 Text Aggregation
    text_treatment_agg_df = aggregate_patient_data(merged_df, KEY_CASE_ID, cols_to_agg_treatment_text)

    # 补充 submitter_id 的帮助函数
    def enrich_submitter_id(target_df, source_map_df):
        if target_df is None: return None
        if KEY_SUBMITTER_ID not in target_df.columns:
            if source_map_df is not None:
                 return pd.merge(source_map_df, target_df, on=KEY_CASE_ID, how='right')
        return target_df

    # 创建 ID 映射表
    if diagnostic_agg_df is not None and KEY_SUBMITTER_ID in diagnostic_agg_df.columns:
        submitter_map_df = diagnostic_agg_df[[KEY_CASE_ID, KEY_SUBMITTER_ID]].drop_duplicates()
    else:
        # 从 merged_df 中提取一个
        submitter_map_df = merged_df[[KEY_CASE_ID, KEY_SUBMITTER_ID]].dropna().drop_duplicates()

    # 确保所有 DF 都有 submitter_id
    treatment_agg_df = enrich_submitter_id(treatment_agg_df, submitter_map_df)
    pathology_agg_df = enrich_submitter_id(pathology_agg_df, submitter_map_df)
    text_treatment_agg_df = enrich_submitter_id(text_treatment_agg_df, submitter_map_df) # 补充 ID

    # 定义要保留的 Key 列
    keys_to_preserve = [KEY_CASE_ID, KEY_SUBMITTER_ID]
    
    # --- 阶段 4: 清洗与过滤 (Cleaning & Filtering) ---
    print("\n[Step 4/5] Processing (Encoding & Filtering)...")
    
    # 待处理的数据集 (需要编码和过滤)
    datasets = {
        'clinical': diagnostic_agg_df,
        'treatment': treatment_agg_df,
        'pathology': pathology_agg_df
    }

    final_results = {}

    # 4.1 处理常规数值/类别数据集
    for name, df in datasets.items():
        if df is None: 
            print(f"Skipping {name} (DataFrame is None).")
            continue
            
        if df.empty:
            print(f"Skipping {name} (DataFrame is empty).")
            continue
            
        print(f"\n>>> Processing {name} dataset ({df.shape[0]} rows, {df.shape[1]} cols)")
        
        # 1. 类型转换与编码
        df_encoded = process_mixed_types_and_encode(df, keys_to_preserve=keys_to_preserve, verbose=True)
        
        # 2. 特征过滤
        df_final = filter_unsupervised_features(
            df_encoded,
            missing_threshold=0.9,
            cardinality_threshold=0.95,
            variance_threshold=0.0,
            verbose=True,
            keep_cols=keys_to_preserve
        )
        
        # 整理列顺序：Key在前
        cols = [c for c in keys_to_preserve if c in df_final.columns] + sorted([c for c in df_final.columns if c not in keys_to_preserve])
        final_results[name] = df_final[cols]

    # 4.2 处理文本数据集 (无需编码和过滤，保留原始文本)
    if text_treatment_agg_df is not None and not text_treatment_agg_df.empty:
        print(f"\n>>> Processing text_treatment dataset ({text_treatment_agg_df.shape[0]} rows)")
        # 仅调整列顺序，不进行编码或过滤
        cols = [c for c in keys_to_preserve if c in text_treatment_agg_df.columns] + \
               sorted([c for c in text_treatment_agg_df.columns if c not in keys_to_preserve])
        
        # 对于 treatments.treatment_type, 默认空值为 Unspecified
        text_treatment_agg_df['treatments.treatment_type'] = text_treatment_agg_df['treatments.treatment_type'].fillna('Unspecified')

        final_results['text_treatment'] = text_treatment_agg_df[cols]
    else:
        print("\n>>> Skipping text_treatment (Empty or None)")

    # --- 阶段 5: 保存结果 ---
    print("\n[Step 5/5] Saving files...")
    
    if 'clinical' in final_results:
        out_path = os.path.join(save_path, 'clinical_data_aggregated.csv')
        final_results['clinical'].to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        
    if 'treatment' in final_results:
        out_path = os.path.join(save_path, 'treatment_data_aggregated.csv')
        final_results['treatment'].to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    if 'pathology' in final_results:
        out_path = os.path.join(save_path, 'pathology_aggregated.csv')
        final_results['pathology'].to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    if 'text_treatment' in final_results: # 新增 text 保存
        out_path = os.path.join(save_path, 'text_treatment.csv')
        final_results['text_treatment'].to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    print("\n✅ Process complete.")

if __name__ == '__main__':
    # 尝试切换到脚本所在目录
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    main()