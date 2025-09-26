import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import gc
import time
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import shutil
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('catalog_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CatalogUpdater:
    """Production-ready catalog vector database updater."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the updater with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.stats = {
            'start_time': None,
            'end_time': None,
            'records_processed': 0,
            'records_added': 0,
            'errors': []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        default_config = {
            "vector_db_path": "catalog_embeddings.pkl",
            "catalog_data_dir": "catalog_data",
            "backup_dir": "Backup",
            "batch_size": 1000,
            "model_name": "all-MiniLM-L6-v2",
            "required_columns": {
                "NAME": "NAME",
                "MANUFACTURER_PRODUCT_NUMBER": "MANUFACTURER_PRODUCT_NUMBER",
                "SUPPLIER_NUMBER": "SUPPLIER_NUMBER",
                "BRAND_NAME": "BRAND_NAME",
                "PACK_SIZE": "PACK_SIZE",
                "GTIN": "GTIN"
            },
            "backup_enabled": True,
            "max_file_size_mb": 5000,
            "memory_cleanup_threshold": 1000
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        else:
            logger.info(f"Config file {config_path} not found. Using default configuration.")
        
        return default_config
    
    def _backup_database(self) -> Optional[str]:
        """Create a backup of the existing database."""
        if not self.config.get("backup_enabled", True):
            return None
            
        backup_dir = Path(self.config["backup_dir"])
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"catalog_embeddings_backup_{timestamp}.pkl"
        
        try:
            if os.path.exists(self.config["vector_db_path"]):
                shutil.copy2(self.config["vector_db_path"], backup_path)
                logger.info(f"Database backed up to: {backup_path}")
                return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            self.stats['errors'].append(f"Backup failed: {e}")
        
        return None
    
    def _validate_file_size(self, file_path: str) -> bool:
        """Validate file size before processing."""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            max_size = self.config.get("max_file_size_mb", 5000)
            
            if file_size_mb > max_size:
                logger.error(f"File size {file_size_mb:.1f}MB exceeds maximum allowed size {max_size}MB")
                return False
            
            logger.info(f"File size validation passed: {file_size_mb:.1f}MB")
            return True
        except Exception as e:
            logger.error(f"Failed to validate file size: {e}")
            return False
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                logger.info(f"Loading model: {self.config['model_name']}")
                self.model = SentenceTransformer(self.config['model_name'])
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self.model
    
    def find_latest_csv_file(self) -> str:
        """Find the latest CSV file in the catalog data directory."""
        directory = Path(self.config["catalog_data_dir"])
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist")
        
        csv_files = list(directory.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory '{directory}'")
        
        # Sort by modification time (newest first)
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"Found {len(csv_files)} CSV files in '{directory}'")
        logger.info(f"Latest file: {latest_file.name}")
        logger.info(f"Modified: {time.ctime(latest_file.stat().st_mtime)}")
        
        return str(latest_file)
    
    def load_vector_database(self) -> Dict[str, List[Any]]:
        """Load the existing vector database."""
        try:
            with open(self.config["vector_db_path"], 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded vector database with {len(data.get('descriptions', []))} records")
            return data
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            raise
    
    def save_vector_database(self, data: Dict[str, List[Any]]) -> None:
        """Save the updated vector database."""
        try:
            with open(self.config["vector_db_path"], 'wb') as f:
                pickle.dump(data, f)
            logger.info("Vector database saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
            raise
    
    def convert_gtin_to_number(self, gtin_str: str) -> Optional[int]:
        """Convert GTIN from scientific notation to regular number."""
        try:
            if pd.isna(gtin_str) or gtin_str == '':
                return None
            gtin_float = float(gtin_str)
            return int(gtin_float)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert GTIN '{gtin_str}': {e}")
            return None
    
    def create_single_embedding_set(self, texts: List[str], text_type: str) -> np.ndarray:
        """Create embeddings for a single set of texts."""
        model = self._load_model()
        batch_size = self.config.get("batch_size", 1000)
        
        total_records = len(texts)
        total_batches = (total_records + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_records:,} {text_type} in {total_batches} batches")
        
        all_embeddings = []
        start_time = time.time()
        
        for i in range(0, total_records, batch_size):
            batch_num = i // batch_size + 1
            end_idx = min(i + batch_size, total_records)
            batch = texts[i:end_idx]
            
            try:
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
                
                # Memory cleanup for large datasets
                if batch_num % self.config.get("memory_cleanup_threshold", 1000) == 0:
                    gc.collect()
                
                logger.debug(f"Batch {batch_num}/{total_batches} completed")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                raise
        
        final_embeddings = np.vstack(all_embeddings)
        processing_time = time.time() - start_time
        
        logger.info(f"Created {text_type} embeddings: {final_embeddings.shape} in {processing_time:.1f}s")
        return final_embeddings
    
    def process_csv_data(self, csv_file_path: str) -> Dict[str, Any]:
        """Process the CSV delta data and extract required columns."""
        logger.info(f"Processing CSV file: {csv_file_path}")
        
        # Validate file size
        if not self._validate_file_size(csv_file_path):
            raise ValueError("File size validation failed")
        
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV file loaded with {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise
        
        # Validate required columns
        required_columns = self.config["required_columns"]
        missing_columns = [col for col in required_columns.keys() if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process data
        processed_data = {}
        
        for db_key, csv_col in required_columns.items():
            if db_key == 'GTIN':
                logger.info("Converting GTIN values from scientific notation")
                gtin_values = [self.convert_gtin_to_number(val) for val in df[csv_col]]
                processed_data['gtins'] = gtin_values
            elif db_key == 'MANUFACTURER_PRODUCT_NUMBER':
                processed_data['mpcs'] = df[csv_col].fillna('').astype(str).tolist()
            elif db_key == 'NAME':
                processed_data['descriptions'] = df[csv_col].fillna('').astype(str).tolist()
            elif db_key == 'SUPPLIER_NUMBER':
                processed_data['supplier_numbers'] = df[csv_col].fillna('').astype(str).tolist()
            elif db_key == 'BRAND_NAME':
                processed_data['brand_names'] = df[csv_col].fillna('').astype(str).tolist()
            elif db_key == 'PACK_SIZE':
                processed_data['pack_sizes'] = df[csv_col].fillna('').astype(str).tolist()
        
        # Add missing columns
        num_rows = len(df)
        processed_data['upcs'] = [None] * num_rows
        processed_data['gtin_450s'] = [None] * num_rows
        processed_data['sources'] = ['Catalog Master Data'] * num_rows
        
        logger.info(f"Processed {num_rows} rows of data")
        
        # Generate embeddings
        logger.info("Creating embeddings for text columns")
        processed_data['desc_embeddings'] = self.create_single_embedding_set(
            processed_data['descriptions'], "descriptions"
        )
        processed_data['brand_embeddings'] = self.create_single_embedding_set(
            processed_data['brand_names'], "brand names"
        )
        processed_data['pack_embeddings'] = self.create_single_embedding_set(
            processed_data['pack_sizes'], "pack sizes"
        )
        
        self.stats['records_processed'] = num_rows
        return processed_data
    
    def update_vector_database(self, csv_file_path: str) -> None:
        """Update the vector database with new data from CSV."""
        try:
            # Create backup
            backup_path = self._backup_database()
            
            # Load existing database
            existing_db = self.load_vector_database()
            
            # Process new data
            new_data = self.process_csv_data(csv_file_path)
            
            # Update database
            logger.info("Updating vector database")
            for key, new_values in new_data.items():
                if key in existing_db:
                    if key in ['desc_embeddings', 'brand_embeddings', 'pack_embeddings']:
                        if hasattr(existing_db[key], 'shape'):
                            existing_db[key] = np.vstack([existing_db[key], new_values])
                        else:
                            existing_db[key] = new_values
                        logger.info(f"Added {new_values.shape[0]} new {key} entries")
                    else:
                        if hasattr(existing_db[key], 'tolist'):
                            existing_db[key] = existing_db[key].tolist()
                        existing_db[key].extend(new_values)
                        logger.info(f"Added {len(new_values)} new {key} entries")
                else:
                    existing_db[key] = new_values
                    logger.info(f"Created new key '{key}' with {len(new_values)} entries")
            
            # Save updated database
            self.save_vector_database(existing_db)
            
            self.stats['records_added'] = self.stats['records_processed']
            logger.info(f"Successfully updated database. Total records: {len(existing_db['descriptions'])}")
            
        except Exception as e:
            logger.error(f"Failed to update vector database: {e}")
            self.stats['errors'].append(str(e))
            raise
    
    def run(self) -> Dict[str, Any]:
        """Main execution method."""
        self.stats['start_time'] = datetime.now()
        
        try:
            logger.info("Starting catalog vector database update")
            
            # Find latest CSV file
            csv_file_path = self.find_latest_csv_file()
            
            # Update database
            self.update_vector_database(csv_file_path)
            
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"Update completed successfully in {duration:.1f} seconds")
            logger.info(f"Records processed: {self.stats['records_processed']}")
            logger.info(f"Records added: {self.stats['records_added']}")
            
            return self.stats
            
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            logger.error(f"Update failed: {e}")
            self.stats['errors'].append(str(e))
            return self.stats

def main():
    """Main function for production execution."""
    try:
        updater = CatalogUpdater()
        stats = updater.run()
        
        # Exit with appropriate code
        if stats['errors']:
            logger.error(f"Update completed with errors: {stats['errors']}")
            exit(1)
        else:
            logger.info("Update completed successfully")
            exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
