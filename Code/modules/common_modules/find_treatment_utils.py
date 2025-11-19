


def find_best_treamtent(model, batch_data, pre_op_cols, post_op_cols, treat_ops):
    # Mask treat_cols
    for col in post_op_cols:
        if col in batch_data:
            batch_data[col] = None

    # Make sure all pretreat_cols exists
    count = 0
    for col in pre_op_cols:
        if col in batch_data and batch_data[col]:
            count += 1
    assert count >= 1, f"Must at least one column except post operation columns"

    # Iterate all treatment options
    batch_size = len(batch_data['pid'])
    batch_min_risk = [1e9 for _ in range(batch_size)]
    batch_best_treat = [None for _ in range(batch_size)]
    for treat in treat_ops:
        assert 'text-treatment' in post_op_cols, f"Modality `texttext-treatment_treatment` must be used as input!"
        batch_data['text-treatment'] = [treat] * batch_size

        output = model(batch_size, batch_data)
        logits = output['logits'].cpu().numpy().tolist()

        for i, risk in enumerate(logits):
            if batch_min_risk[i] > risk[0]:
                batch_min_risk[i] = risk[0]
                batch_best_treat[i] = treat
        
    return batch_min_risk, batch_best_treat



