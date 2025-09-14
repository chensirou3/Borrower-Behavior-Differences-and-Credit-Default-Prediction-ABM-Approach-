"""Main CLI application."""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from acr.config.loader import load_config
from acr.simulation.runner import run_simulation_with_manifest

app = typer.Typer(
    name="acr",
    help="Agent-Based Credit Risk modeling toolkit",
    add_completion=False
)
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def run_sim(
    config: str = typer.Option(
        "configs/experiment.yaml",
        "--config", "-c",
        help="Configuration file path"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output", "-o", 
        help="Output directory (default: auto-generated)"
    ),
    set_params: List[str] = typer.Option(
        [],
        "--set", "-s",
        help="Override config parameters (e.g., --set population.N=1000)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose logging"
    )
) -> None:
    """Run credit risk simulation."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    console.print("[bold blue]🏦 Starting ACR Simulation[/bold blue]")
    
    try:
        # Load configuration
        console.print(f"📋 Loading configuration from {config}")
        cfg = load_config(config, overrides=set_params)
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/run_{timestamp}"
        
        console.print(f"📁 Output directory: {output_dir}")
        
        # Run simulation
        console.print("🚀 Running simulation...")
        
        import numpy as np
        rng = np.random.default_rng(cfg.seed)
        
        result = run_simulation_with_manifest(cfg, output_dir, rng)
        
        # Print summary
        manifest = result['manifest']
        console.print("\n[bold green]✅ Simulation Complete![/bold green]")
        console.print(f"📊 Events generated: {manifest['n_events']:,}")
        console.print(f"👥 Borrowers: {manifest['n_borrowers']:,}")
        console.print(f"📅 Periods: {manifest['n_periods']}")
        console.print(f"⚠️  Default rate: {manifest['default_rate']:.1%}")
        console.print(f"💾 Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Simulation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def eval_models(
    run_dir: str = typer.Argument(..., help="Run directory path"),
    augmented: bool = typer.Option(
        True,
        "--augmented/--baseline",
        help="Evaluate augmented features"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose logging"
    )
) -> None:
    """Evaluate model performance on simulation results."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    console.print("[bold blue]📊 Evaluating Models[/bold blue]")
    
    try:
        # Load results
        events_path = os.path.join(run_dir, "events.csv")
        config_path = os.path.join(run_dir, "config.yaml")
        
        if not os.path.exists(events_path):
            console.print(f"[red]❌ Events file not found: {events_path}[/red]")
            raise typer.Exit(1)
        
        if not os.path.exists(config_path):
            console.print(f"[red]❌ Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"📂 Loading results from {run_dir}")
        
        import pandas as pd
        events_df = pd.read_csv(events_path)
        cfg = load_config(config_path)
        
        console.print(f"📊 Loaded {len(events_df):,} events")
        
        # Build datasets and train models
        from acr.features.builder import build_datasets
        from acr.models.selection import train_test_split_temporal
        from acr.models.pipelines import train_models
        
        console.print("🔧 Building feature datasets...")
        X_baseline, X_augmented, y, group = build_datasets(events_df, cfg)
        
        console.print("✂️  Splitting train/test...")
        (X_train_base, X_test_base, X_train_aug, 
         X_test_aug, y_train, y_test) = train_test_split_temporal(
            X_baseline, X_augmented, y, events_df, cfg
        )
        
        console.print("🤖 Training models...")
        models = train_models(
            X_train_base, X_train_aug, y_train,
            X_test_base, X_test_aug, y_test, cfg
        )
        
        # Simple evaluation (detailed evaluation in separate module)
        console.print("\n[bold green]✅ Model Training Complete![/bold green]")
        
        for model_name, model_info in models.items():
            predictions = model_info['predictions']
            feature_set = model_info['feature_set']
            
            # Basic AUC calculation
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, predictions)
            
            console.print(f"📈 {model_name} ({feature_set}): AUC = {auc:.3f}")
        
        console.print(f"💾 Results available in: {run_dir}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Evaluation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def plots(
    run_dir: str = typer.Argument(..., help="Run directory path"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Plot output directory (default: run_dir)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose logging"
    )
) -> None:
    """Generate comprehensive visualization suite."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    console.print("[bold blue]📊 Generating Comprehensive Visualization Suite[/bold blue]")
    
    try:
        # Load results
        events_path = os.path.join(run_dir, "events.csv")
        config_path = os.path.join(run_dir, "config.yaml")
        
        if not os.path.exists(events_path):
            console.print(f"[red]❌ Events file not found: {events_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"📂 Loading events from {run_dir}")
        
        import pandas as pd
        from acr.config.loader import load_config
        from acr.features.builder import build_datasets
        from acr.models.selection import train_test_split_temporal
        from acr.models.pipelines import train_models
        from acr.viz.plots import create_visualization_suite
        
        # Load data
        events_df = pd.read_csv(events_path)
        console.print(f"📊 Loaded {len(events_df):,} events")
        
        # Load config if available
        if os.path.exists(config_path):
            cfg = load_config(config_path)
        else:
            from acr.config.schema import Config
            cfg = Config()
        
        # Train models for visualization
        console.print("🔧 Building datasets and training models...")
        X_baseline, X_augmented, y, group = build_datasets(events_df, cfg)
        
        (X_train_base, X_test_base, X_train_aug, 
         X_test_aug, y_train, y_test) = train_test_split_temporal(
            X_baseline, X_augmented, y, events_df, cfg
        )
        
        models = train_models(
            X_train_base, X_train_aug, y_train,
            X_test_base, X_test_aug, y_test, cfg
        )
        
        # Set output directory
        if output_dir is None:
            output_dir = run_dir
        
        # Generate visualization suite with quality assurance
        console.print("🎨 Generating visualization suite with quality checks...")
        with console.status("[bold green]Creating plots and tables..."):
            # First run standard validation
            from acr.viz.quality_assurance import run_standard_validation_pipeline
            validation_results = run_standard_validation_pipeline(events_df, models, output_dir)
            
            # Then generate visualizations (now using fixed methods)
            result_paths = create_visualization_suite(
                events_df, models, output_dir
            )
            
            # Add validation results
            result_paths['validation'] = validation_results
        
        # Report results
        console.print("\n[bold green]✅ Visualization Suite Complete![/bold green]")
        console.print(f"📁 Output directory: {output_dir}")
        console.print(f"📊 Generated {len([p for p in result_paths.keys() if 'fig_' in p])} figures")
        console.print(f"📋 Generated {len([p for p in result_paths.keys() if 'tbl_' in p])} tables")
        console.print(f"📝 Summary report: {result_paths.get('summary', 'summary.md')}")
        
        # List key figures
        console.print("\n🖼️  Key Figures Generated:")
        key_figs = [
            'fig_01_roc_overall', 'fig_02_pr_overall', 'fig_03_calibration_overall',
            'fig_04_tradeoff_default', 'fig_05_tradeoff_profit', 'fig_06_heatmap_dti_spendvol'
        ]
        for fig in key_figs:
            if fig in result_paths:
                console.print(f"   📈 {fig}.png")
        
    except Exception as e:
        console.print(f"[bold red]❌ Visualization failed: {e}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def fix_plots(
    run_dir: str = typer.Argument(..., help="Run directory path"),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose logging"
    )
) -> None:
    """Run comprehensive visualization diagnostics and fixes."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    console.print("[bold blue]🔧 Running Visualization Diagnostics & Fixes[/bold blue]")
    
    try:
        # Load results
        events_path = os.path.join(run_dir, "events.csv")
        if not os.path.exists(events_path):
            console.print(f"[red]❌ Events file not found: {events_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"📂 Loading events from {run_dir}")
        
        import pandas as pd
        from acr.config.loader import load_config
        from acr.features.builder import build_datasets
        from acr.models.selection import train_test_split_temporal
        from acr.models.pipelines import train_models
        from acr.viz.diagnostics import VisualizationDiagnostics
        
        # Load data
        events_df = pd.read_csv(events_path)
        console.print(f"📊 Loaded {len(events_df):,} events")
        
        # Load config
        config_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(config_path):
            cfg = load_config(config_path)
        else:
            from acr.config.schema import Config
            cfg = Config()
        
        # Train models for diagnostics
        console.print("🔧 Building datasets and training models...")
        X_baseline, X_augmented, y, group = build_datasets(events_df, cfg)
        
        (X_train_base, X_test_base, X_train_aug, 
         X_test_aug, y_train, y_test) = train_test_split_temporal(
            X_baseline, X_augmented, y, events_df, cfg
        )
        
        models = train_models(
            X_train_base, X_train_aug, y_train,
            X_test_base, X_test_aug, y_test, cfg
        )
        
        # Run diagnostics
        console.print("🔍 Running comprehensive diagnostics...")
        diagnostics = VisualizationDiagnostics(events_df, models, run_dir)
        
        with console.status("[bold green]Running diagnostics and fixes..."):
            results = diagnostics.run_full_diagnostics()
        
        # Report results
        console.print("\n[bold green]✅ Diagnostics & Fixes Complete![/bold green]")
        console.print(f"📁 Fixed outputs in: {run_dir}/figs_fixed/ and {run_dir}/tables_fixed/")
        
        # Show key results
        if 'tradeoff_default' in results:
            assertions = results['tradeoff_default']['assertions']
            console.print(f"📈 Tradeoff Default: Monotonic={'✓' if assertions['monotonic_base'] and assertions['monotonic_aug'] else '✗'}")
            console.print(f"📈 Augmented Better: {'✓' if assertions['aug_better_sufficient'] else '✗'} ({assertions['aug_better_ratio']:.1%})")
        
        if 'regime_curves' in results:
            regime_results = results['regime_curves']['results']
            for regime_name, regime_data in regime_results.items():
                auc_base = regime_data['roc_base'][2]
                auc_aug = regime_data['roc_aug'][2]
                console.print(f"📊 {regime_name.title()} AUC: Base={auc_base:.3f}, Aug={auc_aug:.3f}")
        
        console.print(f"📝 Summary report: {results.get('summary_path', 'summary_fixed.md')}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Diagnostics failed: {e}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def sweep(
    config: str = typer.Option(
        "configs/experiment.yaml",
        "--config", "-c",
        help="Base configuration file"
    ),
    set_params: List[str] = typer.Option(
        [],
        "--set", "-s",
        help="Parameter sweep (e.g., --set population.N=[1000,2000,5000])"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose logging"
    )
) -> None:
    """Run parameter sweep experiments."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    console.print("[bold blue]🔄 Parameter Sweep[/bold blue]")
    console.print("[yellow]⚠️  Parameter sweep will be implemented in next version[/yellow]")
    
    # Placeholder for parameter sweep
    console.print(f"📋 Base config: {config}")
    console.print(f"🎛️  Parameters: {set_params}")


@app.command()
def version() -> None:
    """Show version information."""
    from acr import __version__
    console.print(f"ACR version {__version__}")


if __name__ == "__main__":
    app()
