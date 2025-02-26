import torch

from generate import p2_sampling, ModelWrapper
from path_planning.utils import seed_everything
from transformers import AutoTokenizer, AutoModel


def chat():
    device = 'cuda'
    seed_everything(42)  # For reproducibility
    
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    model_wrapper = ModelWrapper(model)

    gen_length = 128
    num_steps = 100  # Reduced slightly from 128 for efficiency
    mask_id = 126336  # Mask token ID for LLaDA
    
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {num_steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        # Create a tensor with prompt followed by mask tokens
        xt = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        xt[:, :prompt.shape[1]] = prompt.clone()
        
        print(f"Processing response (steps: {num_steps})...")
        
        # Run P2 sampling
        sampled_xt = p2_sampling(
            xt=xt,
            model=model_wrapper,
            mask_id=mask_id,
            num_steps=num_steps,
            tau=0.0,  # Same as temperature=0 in original
            eta=1.0,  # Default value
            kappa_fn=lambda t: t,  # Linear scheduler
            planner=None
        )

        answer = tokenizer.batch_decode(sampled_xt[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # remove the <EOS>
        prompt = sampled_xt[sampled_xt != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    chat()

