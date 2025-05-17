import json
import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

class PipelineDef:
    """
    Model class for pipeline definitions from the database.
    """
    def __init__(self, name: str, source: str = "", class_name: str = "", 
                 custom_pipeline: str = "", default_args: Dict = None, 
                 metadata: Dict = None, components: Dict = None,
                 prompt_def: Optional['PromptDef'] = None, estimated_size_gb: Optional[float] = None):
        self.name = name
        self.source = source
        self.class_name = class_name
        self.custom_pipeline = custom_pipeline
        self.default_args = default_args or {}
        self.metadata = metadata or {}
        self.components = components or {}
        self.prompt_def = prompt_def
        self.estimated_size_gb = estimated_size_gb

class PromptDef:
    """
    Model class for prompt definitions from the database.
    """
    def __init__(self, pipeline_id: int, positive_prompt: str = "", negative_prompt: str = ""):
        self.pipeline_id = pipeline_id
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt

def get_pipeline_defs(db_conn, pipeline_names: List[str]) -> List[PipelineDef]:
    """
    Retrieves pipeline definitions from the database based on their names.
    Similar to the Go GetPipelineDefs function.
    
    Args:
        db_conn: Database connection
        pipeline_names: List of pipeline names to retrieve
        
    Returns:
        List of PipelineDef objects
    """
    if not pipeline_names:
        return []
    
    try:
        pipeline_defs = []
        with db_conn.cursor() as cur:
            # Query to get pipeline definitions with their prompt definitions
            query = """
                SELECT 
                    p.id,
                    p.name,
                    p.source,
                    p.class_name,
                    p.custom_pipeline,
                    p.default_args,
                    p.metadata,
                    p.components,
                    p.estimated_size_bytes,
                    pr.positive_prompt,
                    pr.negative_prompt
                FROM 
                    pipeline_defs p
                LEFT JOIN 
                    prompt_defs pr ON p.prompt_def_id = pr.id
                WHERE 
                    p.name = ANY(%s)
            """
            
            cur.execute(query, (pipeline_names,))
            rows = cur.fetchall()
            
            for row in rows:
                # Create prompt_def if available
                prompt_def = None
                if row['positive_prompt'] or row['negative_prompt']:
                    prompt_def = PromptDef(
                        pipeline_id=row['id'],
                        positive_prompt=row['positive_prompt'] or "",
                        negative_prompt=row['negative_prompt'] or ""
                    )
                
                # Parse JSON fields
                default_args = {}
                if row['default_args']:
                    try:
                        default_args = json.loads(row['default_args']) if isinstance(row['default_args'], str) else row['default_args']
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse default_args for pipeline {row['name']}")
                
                metadata = {}
                if row['metadata']:
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata for pipeline {row['name']}")
                
                components = {}
                if row['components']:
                    try:
                        components = json.loads(row['components']) if isinstance(row['components'], str) else row['components']
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse components for pipeline {row['name']}")

                estimated_size_val_gb = None # Changed name
                if row['estimated_size_bytes'] is not None: # DB column name is still estimated_size_bytes
                    try:
                        if isinstance(row['estimated_size_bytes'], Decimal):
                            estimated_size_val_gb = float(row['estimated_size_bytes'])
                        else:
                            estimated_size_val_gb = float(str(row['estimated_size_bytes']))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert estimated_size_bytes ('{row['estimated_size_bytes']}') to float for {row['name']}: {e}")
                
                # Create PipelineDef
                pipeline_def = PipelineDef(
                    name=row['name'],
                    source=row['source'] or "",
                    class_name=row['class_name'] or "",
                    custom_pipeline=row['custom_pipeline'] or "",
                    default_args=default_args,
                    metadata=metadata,
                    components=components,
                    prompt_def=prompt_def,
                    estimated_size_gb=estimated_size_val_gb
                )
                
                pipeline_defs.append(pipeline_def)
        
        return pipeline_defs
    
    except Exception as e:
        logger.error(f"Error retrieving pipeline definitions: {e}")
        # Ensure transaction is rolled back
        db_conn.rollback()
        raise