#!/usr/bin/env python3
"""
Scatter plot visualization: predicted vs ground truth counts
Visualizes different counting methods (id, line, area) and tracking algorithms (bytetrack, ocsort, botsort, boosttrack)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ground_truth(gt_path):
    """Load ground truth data"""
    try:
        df = pd.read_csv(gt_path)
        df = df.set_index('video_name')
        logger.info(f"Loaded ground truth: {len(df)} videos")
        return df
    except Exception as e:
        logger.error(f"Failed to load ground truth from {gt_path}: {e}")
        sys.exit(1)


def load_predictions(pred_path):
    """Load prediction data"""
    try:
        df = pd.read_csv(pred_path)
        logger.info(f"Loaded predictions from {pred_path}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load predictions from {pred_path}: {e}")
        sys.exit(1)


def calculate_metrics(y_true, y_pred):
    """Calculate R² and RMSE metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if mask.sum() < 2:
        return 0.0, 0.0
    
    r2 = r2_score(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    return r2, rmse


def plot_scatter_grid(gt_df, pred_dfs, output_path):
    """
    Plot scatter grid: counting methods (rows) × tracking algorithms (columns)
    """
    count_methods = ['id', 'line', 'area']
    trackers = ['bytetrack', 'ocsort', 'botsort', 'boosttrack']
    
    # Color scheme for each tracker
    colors = {
        'bytetrack': '#FF8C42',  # Orange
        'ocsort': '#2ECC71',     # Green
        'botsort': '#3498DB',    # Blue
        'boosttrack': '#9B59B6'  # Purple
    }
    
    # Create figure with 3 rows (methods) × 4 columns (trackers)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, method in enumerate(count_methods):
        if method not in pred_dfs:
            logger.warning(f"Skipping method {method}: no prediction data")
            continue
            
        pred_df = pred_dfs[method]
        
        for j, tracker in enumerate(trackers):
            ax = axes[i, j]
            
            # Merge ground truth and predictions
            tracker_data = pred_df[pred_df['tracker'] == tracker].copy()
            merged = tracker_data.merge(
                gt_df[['total']], 
                left_on='video_name', 
                right_index=True, 
                how='inner',
                suffixes=('_pred', '_gt')
            )
            
            if len(merged) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{tracker.capitalize()}', fontsize=14, fontweight='bold')
                continue
            
            # Extract true and predicted values
            y_true = merged['total_gt'].values
            y_pred = merged['total_pred'].values
            
            # Plot scatter points
            ax.scatter(y_true, y_pred, color=colors[tracker], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            # Calculate metrics
            r2, rmse = calculate_metrics(y_true, y_pred)
            
            # Plot ideal prediction line (y=x)
            if len(y_true) > 0:
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                margin = (max_val - min_val) * 0.05
                lim_min = max(0, min_val - margin)
                lim_max = max_val + margin
                
                ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                       'k--', linewidth=2, alpha=0.5, label='Ideal prediction')
                
                # Fit regression line
                if len(y_true) > 1:
                    z = np.polyfit(y_true, y_pred, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(y_true.min(), y_true.max(), 100)
                    ax.plot(x_line, p(x_line), color=colors[tracker], 
                           linewidth=2.5, alpha=0.8, label='Algorithm prediction')
                    
                    # Add regression equation
                    equation = f'y = {z[0]:.2f}x + {z[1]:.2f}'
                    ax.text(0.05, 0.85, equation, transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
            
            # Add metrics text box
            metrics_text = f'$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}'
            ax.text(0.95, 0.15, metrics_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set labels and title
            ax.set_xlabel('Ground truth (GT)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title(f'{tracker.capitalize()}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            
            # Add legend only to first column
            if j == 0:
                ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Add row labels (counting methods)
    for i, method in enumerate(count_methods):
        fig.text(0.02, 0.83 - i * 0.31, method.upper(), 
                fontsize=16, fontweight='bold', rotation=90, 
                va='center', ha='center')
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scatter plot: {output_path}")
    plt.close()


def plot_single_algorithm_comparison(gt_df, pred_dfs, output_dir):
    """
    Plot individual comparison for each counting method and tracker
    Similar to the second reference image style
    """
    count_methods = ['id', 'line', 'area']
    trackers = ['bytetrack', 'ocsort', 'botsort', 'boosttrack']
    
    colors = {
        'bytetrack': '#FF8C42',
        'ocsort': '#2ECC71',
        'botsort': '#3498DB',
        'boosttrack': '#9B59B6'
    }
    
    for method in count_methods:
        if method not in pred_dfs:
            continue
            
        pred_df = pred_dfs[method]
        
        for tracker in trackers:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Merge data
            tracker_data = pred_df[pred_df['tracker'] == tracker].copy()
            merged = tracker_data.merge(
                gt_df[['total']], 
                left_on='video_name', 
                right_index=True, 
                how='inner',
                suffixes=('_pred', '_gt')
            )
            
            if len(merged) == 0:
                plt.close()
                continue
            
            y_true = merged['total_gt'].values
            y_pred = merged['total_pred'].values
            
            # Plot scatter
            ax.scatter(y_true, y_pred, color=colors[tracker], alpha=0.7, 
                      s=100, edgecolors='black', linewidth=0.8)
            
            # Calculate metrics
            r2, rmse = calculate_metrics(y_true, y_pred)
            
            # Plot lines
            if len(y_true) > 0:
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                margin = (max_val - min_val) * 0.05
                lim_min = max(0, min_val - margin)
                lim_max = max_val + margin
                
                # Ideal line
                ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                       'k--', linewidth=2, alpha=0.5, label='Ideal prediction')
                
                # Regression line
                if len(y_true) > 1:
                    z = np.polyfit(y_true, y_pred, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(y_true.min(), y_true.max(), 100)
                    ax.plot(x_line, p(x_line), color=colors[tracker], 
                           linewidth=3, alpha=0.8, label='Algorithm prediction')
                    
                    # Add equation and metrics
                    equation = f'y = {z[0]:.4f}x + {z[1]:.2f}'
                    metrics_text = f'{equation}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f}'
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                
                # Add video labels
                for idx in range(len(y_true)):
                    video_name = merged.iloc[idx]['video_name']
                    ax.annotate(f'Video {idx+1}', (y_true[idx], y_pred[idx]), 
                               textcoords="offset points", xytext=(5, 5),
                               ha='left', fontsize=8, alpha=0.6)
                
                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
            
            ax.set_xlabel('Ground truth (GT)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=14, fontweight='bold')
            ax.set_title(f'{tracker.capitalize()} - {method.upper()}', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
            
            plt.tight_layout()
            output_path = output_dir / f'scatter_{method}_{tracker}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved individual plot: {output_path}")
            plt.close()


def plot_category_tracker_grid(gt_df, pred_df, method, output_path):
    """
    Plot category-tracker grid: categories (rows) × tracking algorithms (columns)
    For a specific counting method
    """
    categories = ['Flower', 'Green', 'Light Purple', 'Blue']
    trackers = ['bytetrack', 'ocsort', 'botsort', 'boosttrack']
    
    # Color scheme for each tracker
    colors = {
        'bytetrack': '#FF8C42',
        'ocsort': '#2ECC71',
        'botsort': '#3498DB',
        'boosttrack': '#9B59B6'
    }
    
    # Create figure with 4 rows (categories) × 4 columns (trackers)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for i, category in enumerate(categories):
        for j, tracker in enumerate(trackers):
            ax = axes[i, j]
            
            # Merge ground truth and predictions
            tracker_data = pred_df[pred_df['tracker'] == tracker].copy()
            merged = tracker_data.merge(
                gt_df[[category]], 
                left_on='video_name', 
                right_index=True, 
                how='inner',
                suffixes=('_pred', '_gt')
            )
            
            if len(merged) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{tracker.capitalize()}', fontsize=14, fontweight='bold')
                continue
            
            # Extract true and predicted values
            y_true = merged[f'{category}_gt'].values
            y_pred = merged[f'{category}_pred'].values
            
            # Plot scatter points
            ax.scatter(y_true, y_pred, color=colors[tracker], alpha=0.7, s=80, 
                      edgecolors='black', linewidth=0.5)
            
            # Calculate metrics
            r2, rmse = calculate_metrics(y_true, y_pred)
            
            # Plot ideal prediction line (y=x)
            if len(y_true) > 0:
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                margin = (max_val - min_val) * 0.05
                lim_min = max(0, min_val - margin)
                lim_max = max_val + margin
                
                ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                       'k--', linewidth=2, alpha=0.5)
                
                # Fit regression line
                if len(y_true) > 1:
                    z = np.polyfit(y_true, y_pred, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(y_true.min(), y_true.max(), 100)
                    ax.plot(x_line, p(x_line), color=colors[tracker], 
                           linewidth=2.5, alpha=0.8)
                
                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
            
            # Add metrics text box
            metrics_text = f'$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}'
            ax.text(0.95, 0.15, metrics_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set labels and title
            ax.set_xlabel('True values', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted values', fontsize=12, fontweight='bold')
            ax.set_title(f'{tracker.capitalize()}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
    
    # Add row labels (categories)
    for i, category in enumerate(categories):
        fig.text(0.02, 0.88 - i * 0.235, category, 
                fontsize=16, fontweight='bold', rotation=90, 
                va='center', ha='center')
    
    plt.suptitle(f'Category-Tracker Comparison ({method.upper()} counting method)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved category-tracker grid: {output_path}")
    plt.close()


def plot_category_single_tracker(gt_df, pred_df, method, tracker, output_path):
    """
    Plot all categories for a specific tracker and counting method
    4 subplots (2×2) for 4 categories
    """
    categories = ['Flower', 'Green', 'Light Purple', 'Blue']
    
    colors = {
        'bytetrack': '#FF8C42',
        'ocsort': '#2ECC71',
        'botsort': '#3498DB',
        'boosttrack': '#9B59B6'
    }
    
    # Create figure with 2×2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        # Merge data
        tracker_data = pred_df[pred_df['tracker'] == tracker].copy()
        merged = tracker_data.merge(
            gt_df[[category]], 
            left_on='video_name', 
            right_index=True, 
            how='inner',
            suffixes=('_pred', '_gt')
        )
        
        if len(merged) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(category, fontsize=16, fontweight='bold')
            continue
        
        y_true = merged[f'{category}_gt'].values
        y_pred = merged[f'{category}_pred'].values
        
        # Plot scatter
        ax.scatter(y_true, y_pred, color=colors[tracker], alpha=0.7, 
                  s=120, edgecolors='black', linewidth=0.8)
        
        # Calculate metrics
        r2, rmse = calculate_metrics(y_true, y_pred)
        
        # Plot lines
        if len(y_true) > 0:
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            margin = (max_val - min_val) * 0.05
            lim_min = max(0, min_val - margin)
            lim_max = max_val + margin
            
            # Ideal line
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                   'k--', linewidth=2, alpha=0.5, label='Ideal prediction')
            
            # Regression line
            if len(y_true) > 1:
                z = np.polyfit(y_true, y_pred, 1)
                p = np.poly1d(z)
                x_line = np.linspace(y_true.min(), y_true.max(), 100)
                ax.plot(x_line, p(x_line), color=colors[tracker], 
                       linewidth=3, alpha=0.8, label='Algorithm prediction')
                
                # Add equation and metrics
                equation = f'y = {z[0]:.3f}x + {z[1]:.2f}'
                metrics_text = f'{equation}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}'
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
        
        ax.set_xlabel('True values', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted values', fontsize=14, fontweight='bold')
        ax.set_title(category, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.suptitle(f'{tracker.capitalize()} - {method.upper()} (All Categories)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved category plot for {tracker}: {output_path}")
    plt.close()


def main():
    """Main function"""
    logger.info("="*80)
    logger.info("Starting scatter plot visualization")
    logger.info("="*80)
    
    # Define paths
    gt_path = Path('dataset/10s_Count_GT.csv')
    pred_dir = Path('output/10s')
    output_dir = Path('output/10s/visual')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load ground truth
    gt_df = load_ground_truth(gt_path)
    
    # Load predictions for all counting methods
    pred_dfs = {}
    for method in ['id', 'line', 'area']:
        pred_file = pred_dir / f'{method}_count.csv'
        if pred_file.exists():
            pred_dfs[method] = load_predictions(pred_file)
        else:
            logger.warning(f"Prediction file not found: {pred_file}")
    
    if not pred_dfs:
        logger.error("No prediction files found")
        sys.exit(1)
    
    # Generate main scatter plot grid (total counts)
    logger.info("Generating main scatter plot grid (total counts)...")
    output_path = output_dir / 'scatter_plot_grid.png'
    plot_scatter_grid(gt_df, pred_dfs, output_path)
    
    # Generate individual comparison plots (total counts)
    logger.info("Generating individual comparison plots (total counts)...")
    plot_single_algorithm_comparison(gt_df, pred_dfs, output_dir)
    
    # Generate category-tracker grids for each counting method
    logger.info("Generating category-tracker grids...")
    for method in ['id', 'line', 'area']:
        if method in pred_dfs:
            output_path = output_dir / f'scatter_category_grid_{method}.png'
            plot_category_tracker_grid(gt_df, pred_dfs[method], method, output_path)
    
    # Generate category plots for each tracker and counting method
    logger.info("Generating category plots for each tracker...")
    trackers = ['bytetrack', 'ocsort', 'botsort', 'boosttrack']
    for method in ['id', 'line', 'area']:
        if method in pred_dfs:
            for tracker in trackers:
                output_path = output_dir / f'scatter_category_{method}_{tracker}.png'
                plot_category_single_tracker(gt_df, pred_dfs[method], method, tracker, output_path)
    
    logger.info("="*80)
    logger.info("Visualization completed successfully")
    logger.info("="*80)


if __name__ == '__main__':
    main()