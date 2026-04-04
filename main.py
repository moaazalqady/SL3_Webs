import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def generate_ssyts_fast(boundary_seq):
    """
    Generates conjugate SSYTs strictly via Pieri's rule branching.
    Operates in O(1) time per valid tableau.
    """
    k = boundary_seq.count('+')
    m = boundary_seq.count('-')
    n = (k + 2 * m) // 3
    L = len(boundary_seq)

    results = []

    # State: index, row1, row2, row3
    def dfs(idx, r1, r2, r3):
        if idx == L:
            results.append((r1, r2, r3))
            return

        val = idx + 1
        sign = boundary_seq[idx]
        l1, l2, l3 = len(r1), len(r2), len(r3)

        if sign == '+':
            # Tensor with V: add one box. Must maintain l1 >= l2 >= l3
            if l1 < n:
                dfs(idx + 1, r1 + [val], r2, r3)
            if l2 < l1:
                dfs(idx + 1, r1, r2 + [val], r3)
            if l3 < l2:
                dfs(idx + 1, r1, r2, r3 + [val])
        else:
            # Tensor with V*: add a vertical domino (two boxes in different rows).
            # Option A: Rows 1 and 2
            if l1 < n and l2 <= l1:
                dfs(idx + 1, r1 + [val], r2 + [val], r3)
            # Option B: Rows 1 and 3
            if l1 < n and l3 < l2:
                dfs(idx + 1, r1 + [val], r2, r3 + [val])
            # Option C: Rows 2 and 3
            if l2 < l1 and l3 <= l2 and l3 < n:
                dfs(idx + 1, r1, r2 + [val], r3 + [val])

    dfs(0, [], [], [])
    return results


def standardize_rows(r1, r2, r3, L):
    """Explodes duplicate labels into a standard 1..3n permutation."""
    pos_dict = {i: [] for i in range(1, L + 1)}
    for c, val in enumerate(r1): pos_dict[val].append((0, c))
    for c, val in enumerate(r2): pos_dict[val].append((1, c))
    for c, val in enumerate(r3): pos_dict[val].append((2, c))

    p = 1
    ST = [[0] * len(r1), [0] * len(r2), [0] * len(r3)]
    for v in range(1, L + 1):
        # Sort top-row first to ensure smaller indices map to higher arcs
        coords = sorted(pos_dict[v], key=lambda x: x[0])
        for r, c in coords:
            ST[r][c] = p
            p += 1
    return ST[0], ST[1], ST[2]


def get_m_diagram_arcs(R1, R2, R3):
    L_arcs, R_arcs = [], []

    avail_R1 = R1.copy()
    for y in R2:
        x = max(val for val in avail_R1 if val < y)
        avail_R1.remove(x)
        L_arcs.append((x, y))

    avail_R2 = R2.copy()
    for z in R3:
        y = max(val for val in avail_R2 if val < z)
        avail_R2.remove(y)
        R_arcs.append((y, z))

    return L_arcs, R_arcs


def clean_graph_fast(G):
    """
    O(V) set-based graph reduction to enforce the non-elliptic property.
    """
    nodes_to_check = set(G.nodes())

    while nodes_to_check:
        node = nodes_to_check.pop()
        if node not in G: continue

        if G.nodes[node].get('type') == 'internal':
            deg = G.degree(node)
            if deg == 2:
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 2:
                    G.add_edge(neighbors[0], neighbors[1])
                    nodes_to_check.add(neighbors[0])
                    nodes_to_check.add(neighbors[1])
                G.remove_node(node)
            elif deg <= 1:
                neighbors = list(G.neighbors(node))
                G.remove_node(node)
                for nbr in neighbors:
                    nodes_to_check.add(nbr)
    return G


def build_web_graph(R1, R2, R3, boundary_seq):
    G = nx.Graph()
    V_dest = {}
    p = 1

    for v, sign in enumerate(boundary_seq, 1):
        b_node = f"B_{v}"
        G.add_node(b_node, type='boundary', label=sign)
        if sign == '+':
            V_dest[p] = b_node
            p += 1
        else:
            y_node = f"Y_minus_{v}"
            G.add_node(y_node, type='internal')
            G.add_edge(b_node, y_node)
            V_dest[p] = y_node
            V_dest[p + 1] = y_node
            p += 2

    L_arcs, R_arcs = get_m_diagram_arcs(R1, R2, R3)

    for y in R2:
        y_node = f"Y_R2_{y}"
        G.add_node(y_node, type='internal')
        G.add_edge(y_node, V_dest[y])

    intersections = []
    for i, (a, b) in enumerate(L_arcs):
        for j, (c, d) in enumerate(R_arcs):
            if (a < c < b < d) or (c < a < d < b):
                xL, rL = (a + b) / 2, (b - a) / 2
                xR, rR = (c + d) / 2, (d - c) / 2
                x_int = (rL ** 2 - rR ** 2 + xR ** 2 - xL ** 2) / (2 * (xR - xL))
                intersections.append({
                    'id': f"k_{i}_{j}", 'x': x_int, 'L_idx': i, 'R_idx': j, 'L_outer': a < c
                })

    L_crossings = {i: [] for i in range(len(L_arcs))}
    R_crossings = {j: [] for j in range(len(R_arcs))}

    for inc in intersections:
        L_crossings[inc['L_idx']].append(inc)
        R_crossings[inc['R_idx']].append(inc)

        k = inc['id']
        Top, Bot = f"Top_{k}", f"Bot_{k}"
        G.add_nodes_from([(Top, {'type': 'internal'}), (Bot, {'type': 'internal'})])
        G.add_edge(Top, Bot)

    for i, (a, b) in enumerate(L_arcs):
        crossings = sorted(L_crossings[i], key=lambda item: item['x'])
        curr = V_dest[a]
        for inc in crossings:
            k = inc['id']
            node_in, node_out = (f"Top_{k}", f"Bot_{k}") if inc['L_outer'] else (f"Bot_{k}", f"Top_{k}")
            G.add_edge(curr, node_in)
            curr = node_out
        G.add_edge(curr, f"Y_R2_{b}")

    for j, (c, d) in enumerate(R_arcs):
        crossings = sorted(R_crossings[j], key=lambda item: item['x'])
        curr = f"Y_R2_{c}"
        for inc in crossings:
            k = inc['id']
            node_in, node_out = (f"Top_{k}", f"Bot_{k}") if not inc['L_outer'] else (f"Bot_{k}", f"Top_{k}")
            G.add_edge(curr, node_in)
            curr = node_out
        G.add_edge(curr, V_dest[d])

    return clean_graph_fast(G)


def planar_layout_tutte(G, boundary_nodes):
    pos = {}
    L = len(boundary_nodes)
    import math

    for i, b_node in enumerate(boundary_nodes):
        angle = math.pi / 2 - (i * 2 * math.pi / L)
        pos[b_node] = (math.cos(angle), math.sin(angle))

    internal_nodes = [n for n in G.nodes() if n not in boundary_nodes]
    if not internal_nodes: return pos

    node_to_idx = {node: i for i, node in enumerate(internal_nodes)}
    n_int = len(internal_nodes)
    A = np.zeros((n_int, n_int))
    bx = np.zeros(n_int)
    by = np.zeros(n_int)

    for node in internal_nodes:
        i = node_to_idx[node]
        neighbors = list(G.neighbors(node))
        A[i, i] = len(neighbors)
        for nbr in neighbors:
            if nbr in boundary_nodes:
                bx[i] += pos[nbr][0]
                by[i] += pos[nbr][1]
            else:
                A[i, node_to_idx[nbr]] = -1

    A += np.eye(n_int) * 1e-8
    x = np.linalg.solve(A, bx)
    y = np.linalg.solve(A, by)

    for node in internal_nodes:
        pos[node] = (x[node_to_idx[node]], y[node_to_idx[node]])
    return pos


def plot_webs(boundary_seq, save_pdf=False):
    k = boundary_seq.count('+')
    m = boundary_seq.count('-')

    if (k + 2 * m) % 3 != 0:
        print(f"Invariant Space Hom({boundary_seq}, C) = 0.")
        return

    L = len(boundary_seq)
    ssyts = generate_ssyts_fast(boundary_seq)

    print(f"Boundary sequence: {boundary_seq}")
    print(f"Dimension of Invariant Space (Number of Webs): {len(ssyts)}")
    if not ssyts: return

    # --- Pagination Setup ---
    cols = 4
    rows_per_page = 5
    webs_per_page = cols * rows_per_page

    if save_pdf:
        filename = f"webs_{boundary_seq}.pdf"
        pdf = PdfPages(filename)
        total_pages = (len(ssyts) + webs_per_page - 1) // webs_per_page
        print(f"Generating multi-page PDF ({total_pages} pages)...")

    for chunk_start in range(0, len(ssyts), webs_per_page):
        chunk = ssyts[chunk_start: chunk_start + webs_per_page]

        # Calculate dynamic rows for the last page if it isn't full
        current_cols = min(cols, len(chunk))
        current_rows = (len(chunk) + current_cols - 1) // current_cols

        fig, axes = plt.subplots(current_rows, current_cols, figsize=(4 * current_cols, 4 * current_rows))

        # Standardize axes into a flat list
        if current_rows == 1 and current_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (r1, r2, r3) in enumerate(chunk):
            ax = axes[idx]
            global_idx = chunk_start + idx + 1

            ST1, ST2, ST3 = standardize_rows(r1, r2, r3, L)
            G = build_web_graph(ST1, ST2, ST3, boundary_seq)

            boundary_nodes = [f"B_{i}" for i in range(1, L + 1)]
            internal_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'internal']

            pos = planar_layout_tutte(G, boundary_nodes)

            circle = plt.Circle((0, 0), 1.0, color='#95a5a6', fill=False, linestyle='--', linewidth=2, zorder=0)
            ax.add_patch(circle)
            ax.set_aspect('equal')

            nx.draw_networkx_nodes(G, pos, nodelist=internal_nodes, node_color='#7ea8d6', node_size=60, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=boundary_nodes, node_color='#2c3e50', node_size=300, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color='#34495e')

            labels = {n: G.nodes[n]['label'] for n in boundary_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_size=12, font_weight='bold', ax=ax)

            ax.set_title(f"Web {global_idx} / {len(ssyts)}", fontsize=10, fontweight='bold')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.axis('off')

        # Hide any unused subplots on the final page
        for i in range(len(chunk), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_pdf:
            pdf.savefig(fig)
            plt.close(fig)  # Critically important: Frees RAM after saving each page
            print(f"  ... Saved webs {chunk_start + 1} to {chunk_start + len(chunk)}")
        else:
            print(f"Showing webs {chunk_start + 1} to {chunk_start + len(chunk)} (Close window to load next batch)")
            plt.show()

    if save_pdf:
        pdf.close()
        print(f"Successfully compiled all webs into {filename}")

if __name__ == "__main__":
    # Test a much larger tensor product (Length 12)
    # The generation and rendering of this invariant space will now execute rapidly.
    plot_webs("++--++--++--", save_pdf=True)