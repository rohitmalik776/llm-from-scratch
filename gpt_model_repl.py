import tiktoken
import torch

import gpt_model


global_settings = {
    'top_k': 5,
    'temperature': 0.8,
}
PROMPT = 'Prompt> '


def get_help():
    return f'''Welcome to the GPT model REPL.
Type help to show this prompt any time.
Type exit to quit the REPL
Type temp <value> to set temperature to the new <value>, CURRENT: {global_settings['temperature']}
Type topk <value> to set top_k to the new <value>, CURRENT: {global_settings['top_k']}'''


def load_model_tokenizer(device):
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = gpt_model.GPTModel(gpt_model.GPT_CONFIG_124M)
    gpt2_params = gpt_model.load_gpt2_params_from_tf_ckpt('./gpt2/124M')
    gpt_model.load_weights_into_gpt(model, gpt2_params)
    del gpt2_params
    model.to(device)

    return model, tokenizer


def generate_sample(model, start_context, tokenizer, device, top_k, temperature):
    token_ids = gpt_model.text_to_token_ids(
        start_context, tokenizer,
    ).to(device)
    context_len = model.pos_embedding.weight.shape[0]

    with torch.no_grad():
        output = gpt_model.generate_text(
            model,
            token_ids,
            50,
            context_len,
            top_k=top_k,
            temperature=temperature,
        )
    output_text = gpt_model.token_ids_to_text(output, tokenizer)

    return output_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_tokenizer(device)
    model.eval()

    print(get_help())
    while True:
        prompt = input(PROMPT)

        if prompt == 'exit':
            exit()
        elif prompt == 'help':
            print(get_help())
            continue
        elif prompt.startswith('topk') and len(prompt.split(' ')) == 2:
            try:
                new_val = int(prompt.split(' ')[-1])
                global_settings['top_k'] = new_val
                print(f'top_k set to {global_settings["top_k"]}')
                continue
            except Exception:
                pass
        elif prompt.startswith('temp') and len(prompt.split(' ')) == 2:
            try:
                new_val = float(prompt.split(' ')[-1])
                global_settings['temperature'] = new_val
                print(f'temperature set to {global_settings["temperature"]}')
                continue
            except Exception:
                pass

        res = generate_sample(
            model,
            prompt,
            tokenizer,
            device,
            global_settings['top_k'],
            global_settings['temperature']
        )
        print(res)


if __name__ == '__main__':
    main()
