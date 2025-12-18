import numpy as np

def get_treatment_risk(model, batch_data, pre_op_cols, post_op_cols, treat_embeds):
    # Make sure all pretreat_cols exists
    count = 0
    for col in pre_op_cols:
        if col in batch_data and batch_data[col]:
            count += 1
    assert count >= 1, f"Must at least one column except post operation columns"

    # Iterate all treatment options
    batch_size = len(batch_data['pid'])
    batch_treatment_risks = [[] for _ in range(batch_size)]
    for idx, treat_embed in enumerate(treat_embeds):
        assert 'text-treatment' in post_op_cols, f"Modality `text-treatment` must be used as input!"
        batch_data['text-treatment'] = [treat_embed] * batch_size

        output = model(batch_size, batch_data)
        logits = output['logits'].cpu().numpy().tolist()

        for i, risk in enumerate(logits):
            batch_treatment_risks[i].append(risk)
        
    return batch_treatment_risks


def get_best_treamtent(treat_ops, treat_onehots, batch_treat_risks):
    assert len(treat_ops) == len(treat_onehots)

    batch_best_treat_ops = []
    batch_best_treat_onehots = []
    for i, risks in enumerate(batch_treat_risks):
        best_idx = np.argmin(risks)
        batch_best_treat_ops.append(treat_ops[best_idx])
        batch_best_treat_onehots.append(treat_onehots[best_idx])

    return batch_best_treat_ops, batch_best_treat_onehots 



