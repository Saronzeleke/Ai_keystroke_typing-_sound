TARGET_SAMPLES_PER_CLASS = 50  # New target

def main():
    print("Place your phone 10–20 cm from the keyboard for clear recordings.")
    create_directory_structure()
    total_samples = 0

    for class_name in CLASSES:
        print(f"\n=== Recording for class '{class_name}' ===")
        class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
        
        # Count how many samples already exist
        existing_files = [
            f for f in os.listdir(class_dir) if f.endswith(".wav")
        ]
        existing_count = len(existing_files)
        remaining = TARGET_SAMPLES_PER_CLASS - existing_count

        print(f"{existing_count} samples found. Need {remaining} more...")

        for idx in range(existing_count, existing_count + remaining):
            filepath = None
            attempts = 0
            max_attempts = 3
            while filepath is None and attempts < max_attempts:
                filepath = record_realtime(class_name, idx)
                attempts += 1
                if filepath is None:
                    print(f"Retry {attempts}/{max_attempts} for {class_name}, sample {idx}")
                    time.sleep(3)
            if filepath:
                total_samples += 1
            else:
                print(f"Failed to record {class_name}, sample {idx} after {max_attempts} attempts")
            time.sleep(3)  # Pause between recordings
        print(f"Completed {TARGET_SAMPLES_PER_CLASS} total samples for '{class_name}'")

    print(f"\nAdditional training data created in {TRAINING_DATA_DIR}")
    print(f"Total new samples recorded: {total_samples}")
