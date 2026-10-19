import os
from matplotlib import pyplot as plt
import seaborn as sns
import squidpy as sq
import scanpy as sc
import spatialdata as sd
import spatialdata_plot
from spatialdata import SpatialData
from anndata import AnnData


def dotplot(
    adata: AnnData,
    markers: list | None = None,
    cluster: str = 'leiden',
    n_genes: int = 5
) -> plt.Figure:
    """
    Generating dotplots showing markers and their expression in the cells.

    This function generates a dotplot by utilizing the Scanpy module.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. It must contain the ranked genes.
    markers : list
        List of the markers shown in the dotplot.
    cluster : str, default = 'leiden'
        Clusters shown in the dotplot.
    n_genes : int, default = 5
        Number of genes to show in the dotplot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated dotplot figure, so it can be saved or displayed
        by the caller (e.g. an agent).

    Notes
    ------
    sc.tl.rank_genes needs to be run first before calling this function.
    """
    if markers is None:
        plot = sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=n_genes,
            show=False,
            return_fig=True,
        )
    else:
        plot = sc.pl.dotplot(
            adata,
            var_names=markers,
            groupby=cluster,
            use_raw=False,
            show=False,
            return_fig=True,
        )

    # DotPlot objects are lazy — this actually builds the axes/figure
    plot.make_figure()
    fig = plot.fig

    return fig

def celltype_composition(
        adata: AnnData,
        condition: str = "condition",
        celltype: str = 'cell_type',
        output_path: str | None = None,
        show: bool = True,
        save: bool = False
):
    """
    Generating a cell type composition graph, showing
    the fractions of cell types per condition(s)

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. It must contain the 
        counts in the data.
    condition : str, default = "condition"
        Column in `adata.obs` containing the conditions
        of the samples.
    celltype : str, default = 'cell_type'
        Column in `adata.obs` containing cell types.
    output_path : str, default = None
        An output path where to write the 
        resulting PNG file.
    show : bool, default = True 
        Show the cell type composition graph
    save : bool, default = False
        Boolean deciding whether or not to save
        the cell type composition graph separately.
    """
    composition_counts = adata.obs.groupby([condition, celltype]).size().reset_index(name='count')
    composition_counts['fraction'] = composition_counts.groupby(condition)['count'].transform(lambda x: x / x.sum())

    # Plot stacked bar chart
    plot_df = composition_counts.pivot(index=condition, columns=celltype, values='fraction').fillna(0)

    plot_df.plot(kind='bar', stacked=True, figsize=(8,5))
    plt.ylabel('Fraction of cells')
    plt.title('Cell type composition per assigned treatment')
    if show:
        plt.legend(title='Cell type', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
    if save:
        plt.tight_layout()
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "celltype_composition.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"The celltype composition figure is saved in the current running directory: {output_path}")