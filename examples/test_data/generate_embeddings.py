import asyncio
import json
from pathlib import Path

from src.task_orchestrator.orchestrator_interface import OrchestratorInterface
from examples.test_data.dialog import chinese_dialogue_natural, english_dialogue_natural, mixed_dialogue_natural


async def generate_embeddings_for_dialog(dialog: list[str], output_filename: str):
    """Generate embeddings for a dialog and save to JSON file."""
    print(f"Processing {len(dialog)} sentences...")

    results = []
    for i, sentence in enumerate(dialog):
        print(f"  {i+1}/{len(dialog)}: {sentence[:50]}...")
        embedding = await OrchestratorInterface.get_embedding(sentence)
        results.append({
            "sentence": sentence,
            "embedding": embedding.tolist()
        })

    output_path = Path(__file__).parent / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")


async def main():
    # Initialize the single process engine with Qwen embedding model
    OrchestratorInterface.initialize("single_process_engine")

    # Generate embeddings for each dialog
    await generate_embeddings_for_dialog(chinese_dialogue_natural, "chinese_dialogue_embeddings.json")
    await generate_embeddings_for_dialog(english_dialogue_natural, "english_dialogue_embeddings.json")
    await generate_embeddings_for_dialog(mixed_dialogue_natural, "mixed_dialogue_embeddings.json")

    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
