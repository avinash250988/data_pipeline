import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
import gc
import time
import os
from pathlib import Path

def load_vector_database(file_path: str) -> Dict[str, List[Any]]:
    """Load the existing 450 vector database from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_vector_database(data: Dict[str, List[Any]], file_path: str) -> None:
    """Save the updated 450 vector database to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def convert_gtin_to_number(gtin_str: str) -> int:
    """Convert GTIN from scientific notation to regular number."""
    try:
        # Convert scientific notation to float first, then to int
        gtin_float = float(gtin_str)
        return int(gtin_float)
    except (ValueError, TypeError):
        return None

def create_single_embedding_set(model, texts, text_type, batch_size=1000):
    """Create embeddings for a single set of texts"""
    total_records = len(texts)
    total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Processing {total_records:,} {text_type} in {total_batches} batches of {batch_size} records each")
    
    all_embeddings = []
    start_time = time.time()
    
    for i in range(0, total_records, batch_size):
        batch_num = i // batch_size + 1
        end_idx = min(i + batch_size, total_records)
        batch = texts[i:end_idx]
        batch_actual_size = len(batch)
        
        batch_start_time = time.time()
        print(f"Batch {batch_num}/{total_batches}: Processing records {i+1:,} to {end_idx:,} ({batch_actual_size} items)...")
        
        # Create embeddings for this batch
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
        
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        
        # Calculate progress and ETA
        progress_pct = (batch_num / total_batches) * 100
        avg_time_per_batch = elapsed_time / batch_num
        eta_seconds = avg_time_per_batch * (total_batches - batch_num)
        eta_minutes = eta_seconds / 60
        
        print(f"  ✓ Batch completed in {batch_time:.1f}s")
        print(f"  📊 Progress: {progress_pct:.1f}% | ETA: {eta_minutes:.1f} minutes")
        print(f"  🕒 Total elapsed: {elapsed_time/60:.1f} minutes")
        print("-" * 60)
        
        # Force garbage collection to free memory
        gc.collect()
    
    print("Combining all batch embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    
    total_time = time.time() - start_time
    print(f"✅ All embeddings created successfully!")
    print(f"📈 Total processing time: {total_time/60:.1f} minutes")
    print(f"📊 Final embedding shape: {final_embeddings.shape}")
    print(f"💾 Memory usage: {final_embeddings.nbytes / (1024*1024):.1f} MB")
    
    return final_embeddings

def create_embeddings_batched(descriptions, batch_size=1000):
    """Create embeddings for descriptions"""
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for descriptions
    print(f"Creating embeddings for {len(descriptions):,} descriptions...")
    print("-" * 60)
    
    # Create description embeddings
    print("📝 Creating description embeddings...")
    desc_embeddings = create_single_embedding_set(model, descriptions, "descriptions", batch_size)
    
    return desc_embeddings

def find_latest_csv_file(directory_path: str) -> str:
    """Find the latest CSV file in the specified directory based on modification time."""
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    
    # Find all CSV files in the directory
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory '{directory_path}'")
    
    # Sort by modification time (newest first)
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    
    print(f"📁 Found {len(csv_files)} CSV files in '{directory_path}'")
    print(f"📅 Latest file: {latest_file.name}")
    print(f"🕒 Modified: {time.ctime(latest_file.stat().st_mtime)}")
    
    return str(latest_file)

def process_csv_data(csv_file_path: str) -> Dict[str, List[Any]]:
    """Process the 450 CSV delta data and extract required columns."""
    print("Reading CSV file...")
    df = pd.read_csv(csv_file_path)
    
    print(f"CSV file loaded with {len(df)} rows")
    print(f"Available columns: {list(df.columns)}")
    
    # Extract required columns - mapping from CSV column names to our database keys
    required_columns = {
        'UniqueID': 'unique_ids',
        'Supplier #': 'supplier_numbers',
        'Supplier Product #': 'mpcs', 
        'UPC': 'upcs',
        'Description': 'descriptions',
        'GTIN': 'gtins',
        'GTIN': 'gtin_450s'  # Note: Both GTIN and 450 GTIN use the same column
    }
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns.keys() if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Process the data
    processed_data = {}
    
    # Process each column
    for csv_col, db_key in required_columns.items():
        if csv_col == 'UniqueID':
            # Process UniqueID column
            print("Processing UniqueID column...")
            processed_data['unique_ids'] = df[csv_col].fillna('').astype(str).tolist()
            print(f"Processed {len(processed_data['unique_ids'])} UniqueID entries")
        elif csv_col == 'GTIN':
            # Convert GTIN from scientific notation to regular numbers
            print("Converting GTIN values from scientific notation...")
            gtin_values = []
            for val in df[csv_col]:
                converted = convert_gtin_to_number(val)
                gtin_values.append(converted)
            
            # Since GTIN appears twice in our mapping, we need to handle both
            if 'gtins' not in processed_data:
                processed_data['gtins'] = gtin_values
            if 'gtin_450s' not in processed_data:
                processed_data['gtin_450s'] = gtin_values
        elif csv_col == 'Supplier #':
            processed_data['supplier_numbers'] = df[csv_col].fillna('').astype(str).tolist()
        elif csv_col == 'Supplier Product #':
            processed_data['mpcs'] = df[csv_col].fillna('').astype(str).tolist()
        elif csv_col == 'UPC':
            processed_data['upcs'] = df[csv_col].fillna('').astype(str).tolist()
        elif csv_col == 'Description':
            processed_data['descriptions'] = df[csv_col].fillna('').astype(str).tolist()
    
    # Add source column
    num_rows = len(df)
    processed_data['sources'] = ['450 Master Data'] * num_rows
    
    print(f"Processed {num_rows} rows of data")
    
    # Generate embeddings for the Description column
    print("\n" + "="*60)
    print("🧠 CREATING EMBEDDINGS FOR NEW DATA")
    print("="*60)
    
    desc_embeddings = create_embeddings_batched(processed_data['descriptions'])
    
    # Add the embeddings to processed data
    processed_data['embeddings'] = desc_embeddings
    
    print("✅ Embeddings created successfully!")
    return processed_data

def update_vector_database(vector_db_path: str, csv_file_path: str) -> None:
    """Update the 450 vector database with new data from CSV."""
    print("Loading existing 450 vector database...")
    existing_db = load_vector_database(vector_db_path)
    
    print("Processing CSV delta data...")
    new_data = process_csv_data(csv_file_path)
    
    print("Updating 450 vector database...")
    # Append new data to existing data
    for key, new_values in new_data.items():
        if key in existing_db:
            # Handle embeddings (numpy arrays) differently
            if key == 'embeddings':
                # For embeddings, concatenate numpy arrays
                if hasattr(existing_db[key], 'shape'):
                    existing_db[key] = np.vstack([existing_db[key], new_values])
                else:
                    existing_db[key] = new_values
                print(f"Added {new_values.shape[0]} new {key} entries (shape: {existing_db[key].shape})")
            else:
                # For other data, convert to list and extend
                if hasattr(existing_db[key], 'tolist'):
                    existing_db[key] = existing_db[key].tolist()
                existing_db[key].extend(new_values)
                print(f"Added {len(new_values)} new {key} entries")
        else:
            existing_db[key] = new_values
            print(f"Created new key '{key}' with {len(new_values) if not hasattr(new_values, 'shape') else new_values.shape[0]} entries")
    
    print(f"Total entries after update: {len(existing_db['descriptions'])}")
    
    # Save the updated database
    print("Saving updated 450 vector database...")
    save_vector_database(existing_db, vector_db_path)
    print("450 vector database updated successfully!")

def main():
    """Main function to update the 450 vector database."""
    vector_db_path = "450_embeddings.pkl"
    csv_file_path = "450_input_data.csv"  # Direct file path since it's in the same directory
    
    try:
        print("="*60)
        print("🚀 450 VECTOR DATABASE UPDATER")
        print("="*60)
        print(f"📄 Using file: {csv_file_path}")
        print("="*60)
        
        # Update the vector database
        update_vector_database(vector_db_path, csv_file_path)
        
        print("="*60)
        print("🎉 450 UPDATE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error updating 450 vector database: {str(e)}")
        raise

if __name__ == "__main__":
    main()
