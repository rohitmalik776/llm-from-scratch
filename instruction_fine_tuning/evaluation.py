import json
from tqdm import tqdm


def safe_parse(text):
    try:
        return json.loads(text)
    except:
        return None


def hallucination_rate(targets, outputs):
    total_unanswerable = 0
    hallucinated = 0

    for t_str, o_str in zip(targets, outputs):
        t = safe_parse(t_str)
        o = safe_parse(o_str)

        if t is None:
            continue

        if t.get("status") == "UNANSWERABLE":
            total_unanswerable += 1

            # hallucination if model answers OR output invalid
            if o is None or o.get("status") != "UNANSWERABLE":
                hallucinated += 1

    return hallucinated / total_unanswerable if total_unanswerable > 0 else 0.0


def failure_rate(targets, outputs):
    total_answerable = 0
    failures = 0

    for t_str, o_str in zip(targets, outputs):
        t = safe_parse(t_str)
        o = safe_parse(o_str)

        if t is None:
            continue

        if t.get("status") == "ANSWERABLE":
            total_answerable += 1

            # failure if model refuses OR invalid output
            if o is None or o.get("status") != "ANSWERABLE":
                failures += 1

    return failures / total_answerable if total_answerable > 0 else 0.0


def json_formatting_rate(targets, outputs):
    valid = 0
    total = len(outputs)

    for o_str in outputs:
        if safe_parse(o_str) is not None:
            valid += 1

    return valid / total if total > 0 else 0.0


def coverage(targets, outputs):
    total_answerable = 0
    correct = 0

    for t_str, o_str in zip(targets, outputs):
        t = safe_parse(t_str)
        o = safe_parse(o_str)

        if t is None:
            continue

        if t.get("status") == "ANSWERABLE":
            total_answerable += 1

            if o is None:
                continue

            if o.get("status") == "ANSWERABLE":
                # compare answers (string match)
                if str(o.get("answer")).strip() == str(t.get("answer")).strip():
                    correct += 1

    return correct / total_answerable if total_answerable > 0 else 0.0


def evaluate_model(targets, outputs, tokenizer, return_txt=False):
    target_txt = []
    output_txt = []

    for target_b, output_b in tqdm(zip(targets, outputs), desc='Evaluating', leave=False):
        b, seq_len = target_b.shape
        for i in range(b):
            target, out = target_b[i, :], output_b[i, :]
            mask = target != -100
            target = target[mask][:-1]
            out = out[mask][:-1]

            target_txt.append(tokenizer.decode(target.tolist()))
            output_txt.append(tokenizer.decode(out.tolist()))

    if return_txt:
        return (
            {
                'hallucination_rate': hallucination_rate(targets=target_txt, outputs=output_txt),
                'failure_rate': failure_rate(targets=target_txt, outputs=output_txt),
                'json_formatting_rate': json_formatting_rate(targets=target_txt, outputs=output_txt),
                'coverage': coverage(targets=target_txt, outputs=output_txt),
            },
            {
                'target_txt': target_txt,
                'output_txt': output_txt,
            },
        )
    return {
        'hallucination_rate': hallucination_rate(targets=target_txt, outputs=output_txt),
        'failure_rate': failure_rate(targets=target_txt, outputs=output_txt),
        'json_formatting_rate': json_formatting_rate(targets=target_txt, outputs=output_txt),
        'coverage': coverage(targets=target_txt, outputs=output_txt),
    }
