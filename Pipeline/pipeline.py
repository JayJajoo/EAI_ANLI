"""
Complete NLI Fine-tuning Pipeline
Runs all stages: EDA -> Training -> Evaluation
"""

import os
import sys
import subprocess
import json
from datetime import datetime
import traceback
from config import Config

# ============================================================================
# STAGE CONFIGURATION
# ============================================================================

# Pipeline stages
STAGES = [
    {
        'name': 'EDA',
        'script': 'eda.py',
        'description': 'Exploratory Data Analysis'
    },
    {
        'name': 'Training',
        'script': 'train.py',
        'description': 'Model Fine-tuning'
    },
    {
        'name': 'Evaluation',
        'script': 'eval.py',
        'description': 'Model Evaluation & Baseline Comparison'
    }
]

# ============================================================================
# LOGGING
# ============================================================================

def log_message(message, console=True):
    """Log to file and optionally console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.PIPELINE_LOG), exist_ok=True)
    
    with open(Config.PIPELINE_LOG, 'a') as f:
        f.write(log_entry + '\n')
    
    if console:
        print(log_entry)

# ============================================================================
# STAGE EXECUTION
# ============================================================================

def run_stage(stage_info):
    """Run a single pipeline stage"""
    stage_name = stage_info['name']
    script_name = stage_info['script']
    description = stage_info['description']
    
    log_message("="*70)
    log_message(f"STAGE: {stage_name}")
    log_message(f"Description: {description}")
    log_message(f"Script: {script_name}")
    log_message("="*70)
    
    # Check if script exists
    if not os.path.exists(script_name):
        error_msg = f"Script not found: {script_name}"
        log_message(error_msg)
        raise FileNotFoundError(error_msg)
    
    start_time = datetime.now()
    log_message(f"Starting at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            # capture_output=True,
            # text=True,
            check=True
        )
        
        # Log output
        if result.stdout:
            log_message("\n--- Stage Output ---")
            log_message(result.stdout)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        log_message(f"\n{stage_name} completed successfully")
        log_message(f"Duration: {duration:.2f} seconds")
        
        return {
            'stage': stage_name,
            'status': 'SUCCESS',
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = f"{stage_name} failed with error:\n{e.stderr}"
        log_message(error_msg)
        
        return {
            'stage': stage_name,
            'status': 'FAILED',
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'error': str(e.stderr)
        }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """Execute complete pipeline"""
    
    pipeline_start = datetime.now()
    
    log_message("\n" + "="*70)
    log_message("NLI FINE-TUNING PIPELINE")
    log_message("="*70)
    log_message(f"Pipeline started: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Total stages: {len(STAGES)}\n")
    
    results = []
    
    # Run each stage
    for i, stage in enumerate(STAGES, 1):
        log_message(f"\n{'#'*70}")
        log_message(f"RUNNING STAGE {i}/{len(STAGES)}: {stage['name']}")
        log_message(f"{'#'*70}\n")
        
        stage_result = run_stage(stage)
        results.append(stage_result)
        
        # Stop pipeline if stage failed
        if stage_result['status'] == 'FAILED':
            log_message(f"\nPipeline stopped due to {stage['name']} failure")
            break
    
    # ========================================================================
    # PIPELINE SUMMARY
    # ========================================================================
    
    pipeline_end = datetime.now()
    total_duration = (pipeline_end - pipeline_start).total_seconds()
    
    log_message("\n" + "="*70)
    log_message("PIPELINE SUMMARY")
    log_message("="*70)
    
    # Check overall status
    all_success = all(r['status'] == 'SUCCESS' for r in results)
    
    summary = {
        'pipeline_status': 'SUCCESS' if all_success else 'FAILED',
        'start_time': pipeline_start.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': pipeline_end.strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration_seconds': total_duration,
        'total_duration_formatted': f"{int(total_duration//60)}m {int(total_duration%60)}s",
        'stages': results
    }
    
    # Log stage results
    for result in results:
        status_symbol = "SUCCESS" if result['status'] == 'SUCCESS' else "FAILED"
        log_message(f"{status_symbol} {result['stage']}: {result['status']} ({result['duration_seconds']:.2f}s)")
    
    log_message(f"\nTotal pipeline duration: {summary['total_duration_formatted']}")
    
    from config import Config
    # Save summary
    with open(Config.PIPELINE_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)
    log_message(f"\nPipeline summary saved to: {Config.PIPELINE_SUMMARY}")
    
    # ========================================================================
    # FINAL STATUS
    # ========================================================================
    
    if all_success:
        log_message("\n" + "="*70)
        log_message("COMPLETE PIPELINE EXECUTED SUCCESSFULLY")
        log_message("="*70)
        
        # Try to load evaluation results
        try:
            from config import Config
            if os.path.exists(Config.EVAL_RESULTS):
                with open(Config.EVAL_RESULTS, 'r') as f:
                    eval_results = json.load(f)
                
                accuracy = eval_results['test_metrics']['accuracy']
                f1_score = eval_results['test_metrics']['f1_macro']
                beats_baseline = eval_results['baseline_comparison']['beats_baseline']
                
                log_message(f"\nFinal Results:")
                log_message(f"  Test Accuracy: {accuracy:.4f}")
                log_message(f"  Test F1 Score: {f1_score:.4f}")
                
                if beats_baseline:
                    log_message(f"\nMODEL BEATS BASELINE")
                    log_message(f"  Baseline: {Config.BASELINE_ACCURACY:.4f} -> Current: {accuracy:.4f}")
                else:
                    log_message(f"\nModel below baseline - further tuning needed")
        except:
            pass
        
        log_message(f"\nAll artifacts saved to: ./artifacts/")
        
        return 0  # Success exit code
    else:
        log_message("\n" + "="*70)
        log_message("PIPELINE FAILED")
        log_message("="*70)
        log_message("Check logs above for error details")
        
        return 1  # Failure exit code

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting Complete NLI Fine-tuning Pipeline")
    print("="*70)
    print("\nStages to execute:")
    for i, stage in enumerate(STAGES, 1):
        print(f"  {i}. {stage['name']}: {stage['description']}")
    print("\n")
    
    try:
        exit_code = run_pipeline()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("\n\nPipeline interrupted by user")
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_msg = f"\nPipeline crashed with error:\n{traceback.format_exc()}"
        log_message(error_msg)
        print(error_msg)
        sys.exit(1)