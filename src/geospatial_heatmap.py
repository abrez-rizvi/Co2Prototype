"""
Geospatial heatmap component for visualizing CO2 emissions across city regions.
Grid-based visualization with North, South, East, West zones and overlapping regions.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional
import streamlit as st


class RegionalGrid:
    """Represents a city divided into regional zones with emission data."""
    
    def __init__(self, grid_size: Tuple[int, int] = (30, 30)):
        """
        Initialize a regional grid.
        
        Args:
            grid_size: (rows, cols) for the grid resolution
        """
        self.rows, self.cols = grid_size
        self.grid = np.zeros((self.rows, self.cols))
        
    def define_regions(self) -> Dict[str, np.ndarray]:
        """
        Define regional masks for North, South, East, West and overlaps.
        
        Returns:
            Dictionary mapping region names to boolean masks
        """
        regions = {}
        
        # Define boundaries (with overlap zones)
        mid_row = self.rows // 2
        mid_col = self.cols // 2
        overlap = self.rows // 6  # Overlap zone size
        
        # North region (top third + overlap)
        north_boundary = mid_row + overlap
        regions['north'] = np.zeros((self.rows, self.cols), dtype=bool)
        regions['north'][:north_boundary, :] = True
        
        # South region (bottom third + overlap)
        south_boundary = mid_row - overlap
        regions['south'] = np.zeros((self.rows, self.cols), dtype=bool)
        regions['south'][south_boundary:, :] = True
        
        # East region (right third + overlap)
        east_boundary = mid_col - overlap
        regions['east'] = np.zeros((self.rows, self.cols), dtype=bool)
        regions['east'][:, east_boundary:] = True
        
        # West region (left third + overlap)
        west_boundary = mid_col + overlap
        regions['west'] = np.zeros((self.rows, self.cols), dtype=bool)
        regions['west'][:, :west_boundary] = True
        
        # Overlapping regions
        regions['north_east'] = regions['north'] & regions['east']
        regions['north_west'] = regions['north'] & regions['west']
        regions['south_east'] = regions['south'] & regions['east']
        regions['south_west'] = regions['south'] & regions['west']
        regions['center'] = regions['north'] & regions['south'] & regions['east'] & regions['west']
        
        return regions
    
    def populate_from_sectors(self, sector_data: Dict[str, float], 
                             sector_to_region_map: Optional[Dict[str, str]] = None):
        """
        Populate grid based on sector emission data.
        
        Args:
            sector_data: Dictionary mapping sector names to emission values
            sector_to_region_map: Optional mapping of sectors to primary regions
                                 (defaults to standard mapping)
        """
        if sector_to_region_map is None:
            # Default mapping: sectors to primary regions
            sector_to_region_map = {
                'transport': 'center',
                'energy': 'north',
                'industry': 'east',
                'residential': 'west',
                'commercial': 'south',
                'waste': 'south_west',
                'agriculture': 'south_east'
            }
        
        regions = self.define_regions()
        
        # Normalize emissions to 0-1 range for intensity
        all_values = list(sector_data.values())
        if len(all_values) == 0:
            return
            
        min_val = min(all_values)
        max_val = max(all_values)
        val_range = max_val - min_val if max_val > min_val else 1
        
        # Fill grid with base noise (representing background emissions)
        self.grid = np.random.uniform(0.1, 0.3, (self.rows, self.cols))
        
        # Add sector emissions to their regions
        for sector, emission in sector_data.items():
            region_key = sector_to_region_map.get(sector, 'center')
            if region_key in regions:
                # Normalized intensity
                intensity = (emission - min_val) / val_range
                
                # Apply intensity to region with some randomness
                mask = regions[region_key]
                regional_variation = np.random.uniform(0.8, 1.2, (self.rows, self.cols))
                self.grid[mask] += intensity * regional_variation[mask]
        
        # Apply some spatial smoothing for realistic gradients
        self._smooth_grid()
        
        # Clip values to valid range
        self.grid = np.clip(self.grid, 0, 2.0)
    
    def _smooth_grid(self, iterations: int = 2):
        """Apply spatial smoothing to create gradients."""
        from scipy.ndimage import gaussian_filter
        self.grid = gaussian_filter(self.grid, sigma=1.5)
    
    def get_grid(self) -> np.ndarray:
        """Return the emission intensity grid."""
        return self.grid


def create_heatmap_figure(grid: np.ndarray, 
                          title: str = "Emission Intensity Map",
                          show_borders: bool = True) -> go.Figure:
    """
    Create a Plotly heatmap figure from emission grid.
    
    Args:
        grid: 2D numpy array of emission intensities
        title: Title for the heatmap
        show_borders: Whether to show regional borders
        
    Returns:
        Plotly Figure object
    """
    rows, cols = grid.shape
    
    # Create custom colorscale (green -> yellow -> orange -> red) with transparency
    colorscale = [
        [0.0, 'rgba(0, 150, 0, 0.5)'],      # Dark green (low) - 50% transparent
        [0.25, 'rgba(100, 200, 0, 0.55)'],   # Light green - 55% transparent
        [0.5, 'rgba(255, 255, 0, 0.6)'],    # Yellow (medium) - 60% transparent
        [0.75, 'rgba(255, 150, 0, 0.65)'],   # Orange - 65% transparent
        [1.0, 'rgba(200, 0, 0, 0.7)']       # Dark red (high) - 70% transparent
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Intensity", side="right"),
            tickmode="linear",
            tick0=0,
            dtick=0.5
        ),
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Intensity: %{z:.2f}<extra></extra>'
    ))
    
    # Add regional border lines if requested
    if show_borders:
        mid_row = rows // 2
        mid_col = cols // 2
        overlap = rows // 6
        
        # Add region boundary lines
        shapes = []
        
        # Horizontal lines (North/South boundary)
        shapes.append(dict(
            type='line',
            x0=0, x1=cols-1,
            y0=mid_row-overlap, y1=mid_row-overlap,
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2, dash='dash')
        ))
        shapes.append(dict(
            type='line',
            x0=0, x1=cols-1,
            y0=mid_row+overlap, y1=mid_row+overlap,
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2, dash='dash')
        ))
        
        # Vertical lines (East/West boundary)
        shapes.append(dict(
            type='line',
            x0=mid_col-overlap, x1=mid_col-overlap,
            y0=0, y1=rows-1,
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2, dash='dash')
        ))
        shapes.append(dict(
            type='line',
            x0=mid_col+overlap, x1=mid_col+overlap,
            y0=0, y1=rows-1,
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2, dash='dash')
        ))
        
        fig.update_layout(shapes=shapes)
    
    # Add region labels as annotations
    mid_row = rows // 2
    mid_col = cols // 2
    
    annotations = [
        dict(x=mid_col, y=rows*0.15, text="NORTH", showarrow=False,
             font=dict(size=10, color='rgba(255, 255, 255, 0.6)')),
        dict(x=mid_col, y=rows*0.85, text="SOUTH", showarrow=False,
             font=dict(size=10, color='rgba(255, 255, 255, 0.6)')),
        dict(x=cols*0.15, y=mid_row, text="WEST", showarrow=False,
             font=dict(size=10, color='rgba(255, 255, 255, 0.6)')),
        dict(x=cols*0.85, y=mid_row, text="EAST", showarrow=False,
             font=dict(size=10, color='rgba(255, 255, 255, 0.6)'))
    ]
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, title="", scaleanchor="x"),
        width=600,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=annotations
    )
    
    return fig


def display_before_after_heatmaps(baseline_sectors: Dict[str, float],
                                   simulated_sectors: Dict[str, float],
                                   grid_size: Tuple[int, int] = (30, 30)):
    """
    Display before and after heatmaps side by side in Streamlit.
    
    Args:
        baseline_sectors: Dictionary of baseline sector emissions
        simulated_sectors: Dictionary of simulated sector emissions
        grid_size: Resolution of the grid
    """
    # Create grids
    baseline_grid = RegionalGrid(grid_size)
    baseline_grid.populate_from_sectors(baseline_sectors)
    
    simulated_grid = RegionalGrid(grid_size)
    simulated_grid.populate_from_sectors(simulated_sectors)
    
    # Create figures
    fig_before = create_heatmap_figure(
        baseline_grid.get_grid(),
        title="Baseline Emission Distribution",
        show_borders=True
    )
    
    fig_after = create_heatmap_figure(
        simulated_grid.get_grid(),
        title="Simulated Emission Distribution",
        show_borders=True
    )
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_before, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_after, use_container_width=True)
    
    # Calculate and display summary statistics
    baseline_total = baseline_grid.get_grid().sum()
    simulated_total = simulated_grid.get_grid().sum()
    change = simulated_total - baseline_total
    pct_change = (change / baseline_total * 100) if baseline_total > 0 else 0
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Total Intensity", f"{baseline_total:.1f}")
    col2.metric("Simulated Total Intensity", f"{simulated_total:.1f}")
    col3.metric("Change", f"{change:+.1f}", f"{pct_change:+.1f}%")


def display_difference_heatmap(baseline_sectors: Dict[str, float],
                               simulated_sectors: Dict[str, float],
                               grid_size: Tuple[int, int] = (30, 30)):
    """
    Display a difference heatmap showing changes between baseline and simulation.
    
    Args:
        baseline_sectors: Dictionary of baseline sector emissions
        simulated_sectors: Dictionary of simulated sector emissions
        grid_size: Resolution of the grid
    """
    # Create grids
    baseline_grid = RegionalGrid(grid_size)
    baseline_grid.populate_from_sectors(baseline_sectors)
    
    simulated_grid = RegionalGrid(grid_size)
    simulated_grid.populate_from_sectors(simulated_sectors)
    
    # Calculate difference
    diff_grid = simulated_grid.get_grid() - baseline_grid.get_grid()
    
    # Create diverging colorscale (blue for decrease, red for increase) with transparency
    colorscale = [
        [0.0, 'rgba(0, 0, 200, 0.6)'],      # Dark blue (large decrease) - 60% transparent
        [0.25, 'rgba(100, 150, 255, 0.5)'], # Light blue - 50% transparent
        [0.5, 'rgba(240, 240, 240, 0.3)'],  # White (no change) - 70% transparent
        [0.75, 'rgba(255, 150, 100, 0.5)'], # Light red - 50% transparent
        [1.0, 'rgba(200, 0, 0, 0.6)']       # Dark red (large increase) - 60% transparent
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=diff_grid,
        colorscale=colorscale,
        zmid=0,  # Center the colorscale at zero
        showscale=True,
        colorbar=dict(
            title=dict(text="Change", side="right")
        ),
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Change: %{z:+.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Emission Change Map (Simulated - Baseline)",
        xaxis=dict(showgrid=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, title="", scaleanchor="x"),
        width=700,
        height=700,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
