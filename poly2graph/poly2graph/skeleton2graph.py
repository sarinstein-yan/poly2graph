import numpy as np
import networkx as nx
from numba import njit
from typing import Optional, Union

# adapted from https://github.com/Image-Py/sknw

def _neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

@njit # mark the array use (0, 1, 2)
def _mark(img, nbs):
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2

@njit # convert index to r, c...
def _idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    rst -= 1
    return rst
    
@njit # fill a node (two or more points)
def _fill(img, p, num, nbs, acc, buf):
    img[p] = num
    buf[0] = p
    cur = 0; s = 1; iso = True
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==2:
                img[cp] = num
                buf[s] = cp
                s+=1
            if img[cp]==1: iso=False
        cur += 1
        if cur==s:break
    return iso, _idx2rc(buf[:s], acc)

@njit # trace the edge and use a buffer, then buf.copy, if using [] numba doesn't work
def _trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0;
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, _idx2rc(buf[:cur+1], acc))
   
@njit # parse the image then get the nodes and edges
def _parse_struc(img, nbs, acc, iso, ring):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64) # 2**17 = 131072
    # buf = np.zeros(1048576, dtype=np.int64) # 2**20 = 1048576
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = _fill(img, p, num, nbs, acc, buf)
            if isiso and not iso: continue
            num += 1
            nodes.append(nds)
    edges = []
    for p in range(len(img)):
        if img[p] <10: continue
        for dp in nbs:
            if img[p+dp]==1:
                edge = _trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    if not ring: return nodes, edges
    for p in range(len(img)):
        if img[p]!=1: continue
        img[p] = num; num += 1
        nodes.append(_idx2rc([p], acc))
        for dp in nbs:
            if img[p+dp]==1:
                edge = _trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges to build a networkx graph
def build_graph(nodes, edges, multi=False, full=True, 
                Potential_image=None, DOS_image=None, add_pts=True):
    
    os = np.array([i.mean(axis=0) for i in nodes])
    if full: os = os.round().astype(np.uint16)

    graph = nx.MultiGraph() if multi else nx.Graph()

    for i in range(len(nodes)):
        if DOS_image is not None:
            node_dos = {'dos': DOS_image[os[i][0], os[i][1]]}
            # dos_list = [DOS_image[pt[0], pt[1]] for pt in nodes[i]]
            # node_dos = {'dos': np.mean(dos_list)}
        else: node_dos = {}

        if Potential_image is not None:
            node_pot = {'potential': Potential_image[os[i][0], os[i][1]]}
        else: node_pot = {}

        graph.add_node(i, o=os[i], **node_dos, **node_pot)

        # if add_pts: graph.nodes[i]['pts'] = nodes[i]

    for s,e,pts in edges:
        if full: pts[[0,-1]] = os[[s,e]]

        if DOS_image is not None:
            dos_list = [DOS_image[pt[0], pt[1]] for pt in pts]
            edge_dos = {'avg_dos': np.mean(dos_list)}
        else: edge_dos = {}

        if Potential_image is not None:
            pot_list = [Potential_image[pt[0], pt[1]] for pt in pts]
            edge_pot = {'avg_potential': np.mean(pot_list)}
        else: edge_pot = {}

        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()

        if add_pts:
            graph.add_edge(s,e, weight=l, pts=pts, **edge_dos, **edge_pot)
        else:
            graph.add_edge(s,e, weight=l, **edge_dos, **edge_pot)

    return graph

def mark_node(ske):
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = _neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    _mark(buf, nbs)
    return buf

def skeleton2graph(
    ske: np.ndarray,
    multi: Optional[bool]=True,
    iso: Optional[bool]=False,
    ring: Optional[bool]=True,
    full: Optional[bool]=True,
    Potential_image: Optional[np.ndarray]=None,
    DOS_image: Optional[np.ndarray]=None,
    add_pts: Optional[bool]=True
) -> Union[nx.Graph, nx.MultiGraph]:
    """
    Converts a skeletonized image into an NetworkX graph object. 

    Parameters:
    -----------
    ske : numpy.ndarray
        The input skeletonized image. This is typically a binary image where
        the skeletonized structures are represented by 1s and the background
        by 0s.
    
    multi : bool, optional, default: True
        If True, the function builds a multi-graph allowing multiple edges 
        between the same set of nodes. If False, only a single edge is 
        allowed between any pair of nodes.
    
    iso : bool, optional, default: False
        If True, isolated nodes (nodes not connected to any other node) 
        are included in the graph. If False, isolated nodes are ignored.
    
    ring : bool, optional, default: True
        If True, the function considers ring structures (closed loops) in 
        the skeleton. If False, ring structures are ignored.
    
    full : bool, optional, default: True
        If True, the graph nodes include the rounded coordinate arrays of the 
        original points. If False, the nodes include the full coordinates.

    Potential_image : numpy.ndarray, optional
        A 2D image of the spectral potential landscape values. If provided,
        the nodes and edges of the graph will include the potential values as
        attributes.

    DOS_image : numpy.ndarray, optional
        A 2D image of the density of states (DOS) values. If provided, the
        nodes and edges of the graph will include the DOS values as attributes.

    add_pts : bool, optional, default: True
        If True, the nodes and edges of the graph will include all original
        points (pixels) that make up the skeleton lines.

    Returns:
    --------
    graph : networkx.Graph or networkx.MultiGraph
        A graph representation of the skeletonized image. Nodes correspond to 
        junction points and endpoints, and edges represent the skeleton lines 
        between them.

    Notes:
    ------
    - The function first pads the input skeleton image to handle edge cases 
      during processing.
    - Neighbors of each pixel in the padded image are calculated to facilitate 
      the conversion of indices.
    - The image is marked using a marking function to classify the points.
    - The marked image is parsed to extract nodes and edges.
    - A NetworkX graph is built from the parsed nodes and edges.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from poly2graph import skeleton2graph
    >>> ske = np.array([
            [0,0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1,1],
            [0,0,0,1,0,0,0,0,0],
            [1,1,1,1,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0],
            [0,1,0,0,0,1,0,0,0],
            [1,0,1,0,0,1,1,1,1],
            [0,1,0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0,0,0]])
    >>> graph = skeleton2graph(ske, multi=True)
    >>> print(graph.nodes)
    >>> print(graph.edges)
    """
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = _neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    _mark(buf, nbs)
    nodes, edges = _parse_struc(buf, nbs, acc, iso, ring)
    return build_graph(nodes, edges, multi, full, 
                       Potential_image, DOS_image, add_pts)

# draw the graph on the skeleton image
def draw_graph_on_skeleton(img, graph, cn=255, ce=128):
    shape = img.shape
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for (s, e) in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]['pts']
                img[np.dot(pts, acc)] = ce
        else: img[np.dot(eds['pts'], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]['pts']
        img[np.dot(pts, acc)] = cn
    return img.reshape(shape)