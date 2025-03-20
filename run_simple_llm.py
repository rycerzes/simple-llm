import argparse
import time
from simple_llm import SimpleLLM


def main():
    parser = argparse.ArgumentParser(description="Run a LLaMA model using SimpleLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name on Hugging Face",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a land far away,",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable detailed debug output"
    )

    args = parser.parse_args()

    llm = SimpleLLM()
    llm.debug = args.debug

    try:
        start_time = time.time()
        llm.load_weights(args.model)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating text...")

        start_time = time.time()
        generated_text = llm.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        generation_time = time.time() - start_time

        print(f"\nGenerated text:\n{generated_text}")
        print(f"\nGeneration completed in {generation_time:.2f} seconds")
        tokens_generated = min(
            args.max_length, len(generated_text.split()) - len(args.prompt.split())
        )
        print(f"Average speed: {tokens_generated / generation_time:.2f} tokens/second")

    except Exception as e:
        import traceback

        print(f"Error: {e}")
        print("\nTraceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
