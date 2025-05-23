import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from poly2graph.skeleton2graph import skeleton2graph
from poly2graph.spectral_graph import contract_close_nodes, add_edges_within_threshold
from skimage.morphology import skeletonize

from numpy.typing import ArrayLike
from typing import Union, Optional, Any

def draw_spectral_graph(
    image: ArrayLike,
    ax: Optional[Any] = None,
    overlay_graph: Optional[bool] = False,
    contract_threshold: Optional[Union[int, float]] = None,
    ax_set_kwargs: Optional[dict] = {},
    s2g_kwargs: Optional[dict] = {}
) -> None:
    """
    Draws the spectral graph over a given image.

    The function displays the image using a specified colormap and,
    optionally, overlays a graph generated from the image's skeleton.
    The overlay includes both node and edge plotting, with the ability
    to contract close nodes based on a threshold.

    Parameters:
        image (ArrayLike): The input image to be processed.
        ax (Optional[Any]): The matplotlib axis to use. If None, the current axis is used.
        overlay_graph (Optional[bool]): If True, compute and overlay the graph on the image.
        contract_threshold (Optional[Union[int, float]]): Threshold to contract close nodes in the graph.
        ax_set_kwargs (Optional[dict]): Additional keyword arguments to pass to ax.set().
        s2g_kwargs (Optional[dict]): Additional keyword arguments for the skeleton2graph function.

    Returns:
        None
    """
    def to_graph(img, **kwargs):
        """
        Converts a binary image into its corresponding graph representation.

        The skeleton of the image is computed and then transformed into a graph
        using the provided skeleton2graph function.

        Parameters:
            img (ArrayLike): Binary image from which the graph is derived.
            **kwargs: Additional arguments to pass to the skeleton2graph function.

        Returns:
            NetworkX Graph: The graph representation of the skeletonized image.
        """
        # Compute skeleton of the image using the Lee method
        ske = skeletonize(img, method='lee')
        # Convert skeleton to a graph representation
        graph = skeleton2graph(ske, add_pts=True, **kwargs)
        return graph

    # If no axis is provided, get the current axis
    if ax is None:
        ax = plt.gca()

    # Display the image on the axis with a grayscale-like 'bone' colormap
    ax.imshow(image, cmap='bone')
    # Set axis labels along with any additional keyword arguments provided
    ax.set(xlabel='Re(E)', ylabel='Im(E)', **ax_set_kwargs)
    # Remove the axis frame for a cleaner presentation
    ax.axis('off')

    # If overlay_graph flag is set to True, compute and draw the graph overlay
    if overlay_graph:
        # Generate the graph overlay with additional parameters (and add points flag set to True)
        overlay_graph = to_graph(image, add_pts=True, **s2g_kwargs)
        # If a contract threshold is provided, contract the nodes in the graph that are close to each other
        if contract_threshold is not None:
            overlay_graph = add_edges_within_threshold(overlay_graph, contract_threshold)
            overlay_graph = contract_close_nodes(overlay_graph, contract_threshold)
        # Loop over each edge in the graph to plot the connecting segments
        for (s, e, key, ps) in overlay_graph.edges(keys=True, data='pts'):
            # Plot edge as a blue line with moderate transparency and thin linewidth
            ax.plot(ps[:,0], ps[:,1], '-', lw=1, alpha=0.8, c='tab:blue')
        # Extract node positions (assumed to be stored under the key 'pos') and plot as red points
        nodes = overlay_graph.nodes()
        ps = np.array([nodes[i]['pos'] for i in nodes])
        ax.scatter(ps[:,0], ps[:,1], c='tab:red', s=5, alpha=0.8)


def mark_graph_skeleton(img, graph, cn=255, ce=128):
    """
    Marks the skeleton graph on a given image by highlighting nodes and edges.

    This function modifies the input image by setting pixel intensities at the
    positions corresponding to the graph's nodes and edges. Nodes are marked with
    a specific intensity (cn) and edges with another (ce).

    Parameters:
        img (ArrayLike): The input image array to be marked.
        graph (NetworkX graph): The graph whose nodes and edges are to be marked on the image.
        cn (int): Intensity value to mark the nodes. Defaults to 255.
        ce (int): Intensity value to mark the edges. Defaults to 128.

    Returns:
        ArrayLike: The marked image, reshaped to its original dimensions.
    """
    # Get the shape of the image and compute an accumulator for flattening indices.
    shape = img.shape
    # The cumulative product is used here to convert multi-dimensional indices to flat indices.
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    # Flatten the image array for easier indexing using a dot product.
    img = img.ravel()

    # Iterate over each edge in the graph to mark the edge pixels
    for (s, e) in graph.edges():
        eds = graph[s][e]
        # If using a MultiGraph, the data is indexed by individual keys
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                # Get the pixels (pts) that represent this edge and mark them with the edge intensity
                pts = eds[i]['pts']
                img[np.dot(pts, acc)] = ce
        else:
            # For a standard graph, directly mark the edge pixels
            img[np.dot(eds['pts'], acc)] = ce

    # Iterate over each node in the graph and mark their corresponding pixels
    for idx in graph.nodes():
        pts = graph.nodes[idx]['pts']
        img[np.dot(pts, acc)] = cn

    # Reshape the flat image back to its original shape before returning
    return img.reshape(shape)
