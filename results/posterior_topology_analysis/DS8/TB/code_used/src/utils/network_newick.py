"""
network_newick.py -- serialisation, validation and structural descriptors of the rooted level-1 networks of
PhyloGFN-Net (version 2; definitions in docs/posterior_topology/MATHEMATICAL_DEFINITIONS.md).

Objects: the `NetNode` DAG of src/env/phylo_network_env.py.  The supported class C is the set of rooted binary
leaf-labelled level-1 networks without parallel edges in which the child of a reticulation is not a reticulation
(the class the sampler generates; it is tree-child).  `network_canonical_string`, `all_nodes`, `reticulation_nodes`
and `displayed_trees` are the project's own functions and are imported, not re-implemented.

Contents
 1. Extended Newick writer / parser (Rich-Newick `:length:support:gamma` fields on hybrid edges) and a second,
    independent serialisation (edge record) used to validate the first.
 2. validate_network: acyclicity, degrees, parent-slot consistency, leaf set, parameter domains, level-1, tree-child.
 3. Root-equivalence: the undirected multigraph with the degree-2 root suppressed, the root positions that give a
    network of C again ("in-class rootings"), re-rooting, and the canonical id of the root-equivalence class
    (semi_directed_topology_id = smallest rooted canonical string over the in-class rootings).
 4. Displayed trees as unrooted split sets, their inheritance-weighted topology distribution, Robinson-Foulds.
 5. Reticulation (cycle) signatures: hybrid block + circular sequence of the pendant taxon blocks around the cycle.
 6. Root-invariant taxon-based keys for every edge and every inheritance probability (parameter matching).
 7. mu-representation (path-count vectors) of Cardona, Rossello & Valiente (2009) for the mu-distance.
"""
import json
import math
import re
from collections import deque, Counter

from src.env.phylo_network_env import NetNode, all_nodes, reticulation_nodes, network_canonical_string
from src.utils.network_likelihood import displayed_trees

NEWICK_RESERVED = set(" \t\n\r()[]':;,")
_NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')
_HYBRID_TAG = re.compile(r'^#([A-Za-z]*)(\d+)$')


# ============================================================================ 1. labels, numbers, writer
def quote_label(name):
    """Newick label.  Single-quoted (embedded quotes doubled) when it is empty, contains a reserved character or
    contains '#' (an unquoted '#' starts a hybrid tag in Extended Newick)."""
    name = str(name)
    if name == '' or '#' in name or any(ch in NEWICK_RESERVED for ch in name):
        return "'" + name.replace("'", "''") + "'"
    return name


def _fmt(x):
    """shortest decimal string that reads back to exactly the same float64"""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError('non-finite number %r cannot be serialised' % x)
    return repr(x)


def to_enewick(root, taxa, hybrid_prefix='H'):
    """Extended Newick string of the rooted network `root`; `taxa[i]` is the label of leaf i.

    (subtree)#Hk:length::gamma at the first parent met in the depth-first traversal, #Hk:length::gamma at the
    second; `length` = length of that parent edge, `gamma` = its inheritance probability (theta for slot 0,
    1 - theta for slot 1); tree edges `:length`; no length on the root.  Reticulations are numbered in order of
    first appearance.  A network without reticulations is a plain Newick tree."""
    labels = {}

    def gamma_of(h, parent):
        if h.parents[0] is parent:
            return h.theta
        if h.parents[1] is parent:
            return 1.0 - h.theta
        raise ValueError('node %d is not a parent of reticulation %d' % (parent.idx, h.idx))

    def rec(v, parent, length):
        tail = '' if length is None else ':' + _fmt(length)
        if v.kind == 'leaf':
            return quote_label(taxa[v.idx]) + tail
        if v.kind == 'ret':
            g = '::' + _fmt(gamma_of(v, parent))
            if v.idx not in labels:
                labels[v.idx] = '#%s%d' % (hybrid_prefix, len(labels) + 1)
                inner = rec(v.children[0], v, v.child_dists[0])
                return '(' + inner + ')' + labels[v.idx] + tail + g
            return labels[v.idx] + tail + g
        return '(' + ','.join(rec(c, v, d) for c, d in zip(v.children, v.child_dists)) + ')' + tail
    return rec(root, None, None) + ';'


# ============================================================================ 1b. parser
class _Tok(object):
    def __init__(self, s):
        self.s, self.i = s, 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ''

    def take(self):
        ch = self.peek()
        self.i += 1
        return ch

    def skip(self):
        """skip whitespace and [comments]"""
        while True:
            while self.peek() != '' and self.peek() in ' \t\n\r':
                self.i += 1
            if self.peek() == '[':
                j = self.s.find(']', self.i)
                if j < 0:
                    raise ValueError('unterminated comment')
                self.i = j + 1
                continue
            return

    def read_label(self):
        """returns (text, quoted)"""
        self.skip()
        if self.peek() == "'":
            self.take()
            out = []
            while True:
                ch = self.take()
                if ch == '':
                    raise ValueError('unterminated quoted label')
                if ch == "'":
                    if self.peek() == "'":
                        self.take(); out.append("'")
                    else:
                        return ''.join(out), True
                else:
                    out.append(ch)
        out = []
        while self.peek() != '' and self.peek() not in NEWICK_RESERVED:
            out.append(self.take())
        return ''.join(out), False

    def read_number(self):
        """a finite decimal number, or None when the field is empty"""
        self.skip()
        m = _NUMBER.match(self.s, self.i)
        if not m:
            if self.peek() not in (':', ',', ')', ';', ''):
                raise ValueError('malformed number at position %d: %r' % (self.i, self.s[self.i:self.i + 12]))
            return None
        self.i = m.end()
        if self.peek() != '' and self.peek() not in ':,); \t\n\r[':
            raise ValueError('malformed number at position %d: %r' % (m.start(), self.s[m.start():self.i + 6]))
        x = float(m.group(0))
        if not math.isfinite(x):
            raise ValueError('non-finite number %r' % m.group(0))
        return x

    def read_attrs(self):
        """':length:support:gamma' with any field empty -> (length, support, gamma)"""
        vals = [None, None, None]
        k = 0
        self.skip()
        while self.peek() == ':':
            if k == 3:
                raise ValueError('more than three ":" fields at position %d' % self.i)
            self.take()
            vals[k] = self.read_number()
            k += 1
            self.skip()
        return tuple(vals)


def parse_enewick(string, taxa, require_gamma=True, validate=True):
    """Parse a plain or extended Newick string into a rooted NetNode DAG of the supported class.

    returns (root, info), info = {'n_hybrids', 'hybrids': {index: node}, 'gamma_inferred': [indices]}.

    Accepted: quoted and unquoted labels, [comments], internal node labels (ignored), hybrid tags `#<letters><int>`
    (unquoted only; the integer identifies the node, a name may precede the tag), `:length:support:gamma` with
    empty fields.  Refused with ValueError: unknown, missing or repeated taxa; a quoted label is never a hybrid tag;
    a reticulation that does not occur exactly twice with exactly one subtree of exactly one child; a gamma outside
    [0, 1], gammas not summing to 1 (tolerance 1e-6), a gamma on a non-hybrid edge, no gamma on a reticulation when
    require_gamma; negative, non-finite or malformed numbers; non-binary tree nodes; text after ';'; and, with
    validate=True, every structural violation listed in validate_network (cycles, level > 1, parallel edges, ...).
    The parent written first fills slot 0 (theta = its gamma)."""
    taxon_index = {str(t): i for i, t in enumerate(taxa)}
    if len(taxon_index) != len(taxa):
        raise ValueError('duplicate taxon names')
    tok = _Tok(string.strip())
    next_idx = [len(taxa)]
    hybrids = {}             # integer -> {'node', 'defined', 'gammas': [gamma of occurrence 0, of occurrence 1], 'occurrences'}
    seen_leaves = set()

    def new_node(kind):
        v = NetNode(next_idx[0], kind)
        next_idx[0] += 1
        return v

    def attach(parent, child, length, gamma, tag):
        if length is not None and length < 0:
            raise ValueError('negative branch length %r on the edge into node %d' % (length, child.idx))
        d = 0.0 if length is None else float(length)
        parent.children.append(child)
        parent.child_dists.append(d)
        parent.child_slots.append(None)
        if child.kind == 'ret':
            if gamma is not None and not (0.0 <= gamma <= 1.0):
                raise ValueError('inheritance probability %r of reticulation #%d is outside [0, 1]' % (gamma, tag))
            if child.parents[0] is None:
                child.parents[0] = parent; child.parent_dists[0] = d
            elif child.parents[1] is None:
                child.parents[1] = parent; child.parent_dists[1] = d
            else:
                raise ValueError('reticulation #%d has more than two parent edges' % tag)
            hybrids[tag]['gammas'].append(gamma)

            slot = next(
                (k for k in (0, 1) if child.parents[k] is parent),
                None
            )
            if slot is None:
                raise ValueError("reticulation parent-slot mismatch")
            parent.child_slots[-1] = slot
        elif gamma is not None:
            raise ValueError('inheritance probability on a non-hybrid edge (into node %d)' % child.idx)

    def parse_subtree():
        """returns (node, length_from_parent, gamma_or_None, hybrid_index_or_None)"""
        tok.skip()
        children = None
        if tok.peek() == '(':
            tok.take()
            children = []
            while True:
                children.append(parse_subtree())
                tok.skip()
                ch = tok.take()
                if ch == ',':
                    continue
                if ch == ')':
                    break
                raise ValueError('expected , or ) at position %d' % tok.i)
        label, quoted = tok.read_label()
        length, support, gamma = tok.read_attrs()
        tag = None
        if not quoted and '#' in label:
            name, _, rest = label.partition('#')
            m = _HYBRID_TAG.match('#' + rest)
            if not m:
                raise ValueError('malformed hybrid tag %r' % label)
            tag = int(m.group(2))
            label = name
        if tag is not None:
            entry = hybrids.setdefault(tag, {'node': None, 'defined': False, 'gammas': [], 'occurrences': 0})
            if entry['node'] is None:
                entry['node'] = new_node('ret')
            entry['occurrences'] += 1
            if entry['occurrences'] > 2:
                raise ValueError('reticulation #%d occurs more than twice' % tag)
            node = entry['node']
            if children is not None:
                if entry['defined']:
                    raise ValueError('reticulation #%d has its subtree written twice' % tag)
                if len(children) != 1:
                    raise ValueError('reticulation #%d must have exactly one child (found %d)' % (tag, len(children)))
                c, clen, cg, ctag = children[0]
                attach(node, c, clen, cg, ctag)
                node.child_dist = node.child_dists[0]
                entry['defined'] = True
            return node, length, gamma, tag
        if children is not None:
            if len(children) != 2:
                raise ValueError('tree node with %d children (binary required)' % len(children))
            node = new_node('tree')
            for c, clen, cg, ctag in children:
                attach(node, c, clen, cg, ctag)
            return node, length, gamma, None
        if label not in taxon_index:
            raise ValueError('unknown taxon label %r' % label)
        if label in seen_leaves:
            raise ValueError('taxon %r occurs twice' % label)
        seen_leaves.add(label)
        return NetNode(taxon_index[label], 'leaf'), length, gamma, None

    root, root_len, root_gamma, root_tag = parse_subtree()
    tok.skip()
    if tok.take() != ';':
        raise ValueError('missing terminal ;')
    tok.skip()
    if tok.peek() != '':
        raise ValueError('text after the terminal ;')
    if root_tag is not None or root.kind != 'tree':
        raise ValueError('the root must be a binary tree node')
    if root_gamma is not None:
        raise ValueError('inheritance probability on the root')
    inferred = []
    for tag, entry in sorted(hybrids.items()):
        h = entry['node']
        if not entry['defined']:
            raise ValueError('reticulation #%d has no subtree' % tag)
        if entry['occurrences'] != 2 or h.parents[1] is None:
            raise ValueError('reticulation #%d must occur exactly twice (found %d)' % (tag, entry['occurrences']))
        g0, g1 = entry['gammas']
        if g0 is None and g1 is None:
            if require_gamma:
                raise ValueError('reticulation #%d has no inheritance probability' % tag)
            h.theta = None
        elif g0 is None or g1 is None:
            h.theta = float(g0) if g0 is not None else 1.0 - float(g1)
            inferred.append(tag)
        else:
            if abs(g0 + g1 - 1.0) > 1e-6:
                raise ValueError('inheritance probabilities of reticulation #%d sum to %r, not 1' % (tag, g0 + g1))
            h.theta = float(g0)
    for k, tag in enumerate(sorted(hybrids)):
        hybrids[tag]['node'].ret_id = k
    if seen_leaves != set(taxon_index):
        raise ValueError('taxa missing from the string: %s' % sorted(set(taxon_index) - seen_leaves)[:5])
    if validate:
        validate_network(root, len(taxa), require_params=require_gamma)
    assign_tokens(root)
    return root, {'n_hybrids': len(hybrids), 'hybrids': {k: v['node'] for k, v in hybrids.items()}, 'gamma_inferred': inferred}


def assign_tokens(root):
    """node.tokens = frozenset of the leaf indices below the node (iterative, children before parents)"""
    order = _topological_order(root)
    for v in reversed(order):
        v.tokens = frozenset([v.idx]) if v.kind == 'leaf' else frozenset().union(*[c.tokens for c in v.children])


# ============================================================================ 1c. independent serialisation
def to_edge_record(root):
    """A second serialisation, written directly from the NetNode fields (no traversal logic shared with the Newick
    writer): {'root', 'nodes': [[idx, kind]], 'edges': [[parent, child, length, gamma|None, slot|None]]}; the edges
    of every node are listed in the order of its `children`."""
    nodes, edges = [], []
    for v in all_nodes(root):
        nodes.append([int(v.idx), v.kind])
        for c, d in zip(v.children, v.child_dists):
            gamma, slot = None, None
            if c.kind == 'ret':
                slot = 0 if c.parents[0] is v else 1
                if c.parents[slot] is not v:
                    raise ValueError('reticulation %d does not list node %d as a parent' % (c.idx, v.idx))
                gamma = float(c.theta) if slot == 0 else 1.0 - float(c.theta)
            edges.append([int(v.idx), int(c.idx), float(d), gamma, slot])
    return {'root': int(root.idx), 'nodes': nodes, 'edges': edges}


def from_edge_record(rec, n_leaves, validate=True):
    """rebuild the NetNode DAG (with parent slots and theta) from an edge record"""
    node = {int(i): NetNode(int(i), kind) for i, kind in rec['nodes']}
    if len(node) != len(rec['nodes']):
        raise ValueError('duplicate node index in the edge record')
    gam = {}
    for p, c, d, gamma, slot in rec['edges']:
        P, Cn = node[int(p)], node[int(c)]
        P.children.append(Cn)
        P.child_dists.append(float(d))
        if Cn.kind == 'ret':
            if slot not in (0, 1) or Cn.parents[slot] is not None:
                raise ValueError('bad parent slot for reticulation %d' % Cn.idx)
            Cn.parents[slot] = P
            Cn.parent_dists[slot] = float(d)
            gam[(Cn.idx, slot)] = gamma
        elif gamma is not None or slot is not None:
            raise ValueError('inheritance data on a non-hybrid edge (%d, %d)' % (p, c))
    k = 0
    for i in sorted(node):
        v = node[i]
        if v.kind == 'ret':
            g0, g1 = gam.get((i, 0)), gam.get((i, 1))
            if g0 is None or g1 is None or abs(g0 + g1 - 1.0) > 1e-9:
                raise ValueError('reticulation %d: inheritance probabilities %r, %r' % (i, g0, g1))
            v.theta = float(g0)
            if len(v.children) == 1:
                v.child_dist = v.child_dists[0]
            v.ret_id = k
            k += 1
    root = node[int(rec['root'])]
    if validate:
        validate_network(root, n_leaves)
    assign_tokens(root)
    return root


def network_from_canonical_string(text, n_leaves, length=1.0, theta=0.5):
    """Rooted network of the supported class from its canonical string (the format of network_canonical_string:
    'L<i>' leaves, '(a,b)' tree nodes, '#x' the reticulation with subtree x written at both parents), with every
    branch length = `length` and every inheritance probability = `theta`.  Topology-only constructor used to attach
    structure to the keys of an enumerated reference and to build test cases."""
    pos = [0]

    def ast():
        ch = text[pos[0]]
        if ch == 'L':
            j = pos[0] + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            node = ('L', int(text[pos[0] + 1:j]))
            pos[0] = j
            return node
        if ch == '#':
            start = pos[0] + 1
            pos[0] = start
            child = ast()
            return ('#', text[start:pos[0]], child)
        if ch == '(':
            pos[0] += 1
            a = ast()
            if text[pos[0]] != ',':
                raise ValueError('malformed canonical string at position %d' % pos[0])
            pos[0] += 1
            b = ast()
            if text[pos[0]] != ')':
                raise ValueError('malformed canonical string at position %d' % pos[0])
            pos[0] += 1
            return ('T', a, b)
        raise ValueError('malformed canonical string at position %d' % pos[0])
    tree = ast()
    if pos[0] != len(text):
        raise ValueError('text after the canonical string')
    next_idx = [n_leaves]
    rets = {}

    def build(a, parent):
        if a[0] == 'L':
            node = NetNode(a[1], 'leaf')
        elif a[0] == '#':
            if a[1] in rets:
                node = rets[a[1]]
            else:
                node = NetNode(next_idx[0], 'ret'); next_idx[0] += 1
                rets[a[1]] = node
                node.theta = float(theta)
                c = build(a[2], node)
                node.children = [c]; node.child_dists = [float(length)]; node.child_dist = float(length)
            slot = 0 if node.parents[0] is None else 1
            if node.parents[slot] is not None:
                raise ValueError('a reticulation occurs more than twice in the canonical string')
            node.parents[slot] = parent
            node.parent_dists[slot] = float(length)
        else:
            node = NetNode(next_idx[0], 'tree'); next_idx[0] += 1
            for sub in (a[1], a[2]):
                node.children.append(build(sub, node))
                node.child_dists.append(float(length))
        return node
    root = build(tree, None)
    for k, key in enumerate(sorted(rets)):
        rets[key].ret_id = k
    validate_network(root, n_leaves)
    assign_tokens(root)
    return root


# ============================================================================ 2. validation
def _topological_order(root):
    """nodes reachable from root in a topological order (parents first); ValueError on a directed cycle"""
    nodes = all_nodes(root)                               # iterative, terminates on cyclic input
    by_id = {}
    for v in nodes:
        by_id[v.idx] = v
    indeg = {v.idx: 0 for v in nodes}
    for v in nodes:
        for c in v.children:
            if by_id.get(c.idx) is not c:
                raise ValueError('invalid network: two nodes share the index %d' % c.idx)
            indeg[c.idx] += 1
    if indeg[root.idx] != 0:
        raise ValueError('invalid network: directed cycle through the root')
    queue, order = deque([root]), []
    left = dict(indeg)
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in v.children:
            left[c.idx] -= 1
            if left[c.idx] == 0:
                queue.append(c)
    if len(order) != len(nodes):
        raise ValueError('invalid network: directed cycle (a node is its own ancestor)')
    return order


def validate_network(root, n_leaves, require_params=True):
    """ValueError unless `root` is a network of the supported class C with valid parameters:
    acyclic; unique node indices; root = tree node of in-degree 0; leaves = exactly the indices 0..n_leaves-1,
    in-degree 1, no children; tree nodes: in-degree 1 (root 0) and two DISTINCT children; reticulations: in-degree 2
    from two distinct parents recorded consistently in parents[0], parents[1] (with parent_dists equal to the
    parents' child_dists), one child that is not a reticulation; branch lengths finite and >= 0, theta finite in
    [0, 1] (require_params); cycles pairwise vertex-disjoint with no other reticulation on them (level-1);
    tree-child (every internal node has a non-reticulation child).  Returns the nodes in topological order."""
    order = _topological_order(root)
    if root.kind != 'tree':
        raise ValueError('invalid network: the root is not a tree node')
    indeg = Counter()
    for v in order:
        if v.kind not in ('leaf', 'tree', 'ret'):
            raise ValueError('invalid network: node %d has unknown kind %r' % (v.idx, v.kind))
        if len(v.children) != len(v.child_dists):
            raise ValueError('invalid network: node %d has %d children but %d lengths' % (v.idx, len(v.children), len(v.child_dists)))
        for c, d in zip(v.children, v.child_dists):
            indeg[c.idx] += 1
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d < 0:
                raise ValueError('invalid network: branch length %r on edge (%d, %d)' % (d, v.idx, c.idx))
        if v.kind == 'leaf' and v.children:
            raise ValueError('invalid network: leaf %d has children' % v.idx)
        if v.kind == 'tree':
            if len(v.children) != 2:
                raise ValueError('invalid network: tree node %d has %d children' % (v.idx, len(v.children)))
            if v.children[0] is v.children[1]:
                raise ValueError('invalid network: parallel edges from node %d to node %d' % (v.idx, v.children[0].idx))
            if all(c.kind == 'ret' for c in v.children):
                raise ValueError('invalid network: both children of node %d are reticulations (not tree-child)' % v.idx)
        if v.kind == 'ret':
            if len(v.children) != 1:
                raise ValueError('invalid network: reticulation %d has %d children' % (v.idx, len(v.children)))
            if v.children[0].kind == 'ret':
                raise ValueError('invalid network: the child of reticulation %d is a reticulation' % v.idx)
    leaves = sorted(v.idx for v in order if v.kind == 'leaf')
    if leaves != list(range(n_leaves)):
        raise ValueError('invalid network: leaf indices %s..., expected exactly 0..%d' % (leaves[:8], n_leaves - 1))
    for v in order:
        want = 0 if v is root else (2 if v.kind == 'ret' else 1)
        if indeg[v.idx] != want:
            raise ValueError('invalid network: node %d (%s) has in-degree %d, expected %d' % (v.idx, v.kind, indeg[v.idx], want))
        if v.kind == 'ret':
            p0, p1 = v.parents
            if p0 is None or p1 is None or p0 is p1:
                raise ValueError('invalid network: reticulation %d does not have two distinct recorded parents' % v.idx)
            for k, p in enumerate((p0, p1)):
                pos = [t for t, c in enumerate(p.children) if c is v]
                if len(pos) != 1:
                    raise ValueError('invalid network: parent slot %d of reticulation %d is inconsistent' % (k, v.idx))
                if p.child_dists[pos[0]] != v.parent_dists[k]:
                    raise ValueError('invalid network: parent_dists[%d] of reticulation %d differs from the parent edge' % (k, v.idx))
            if v.child_dist is None or v.child_dist != v.child_dists[0]:
                raise ValueError('invalid network: child_dist of reticulation %d is inconsistent' % v.idx)
            if require_params and (v.theta is None or not math.isfinite(v.theta) or not (0.0 <= v.theta <= 1.0)):
                raise ValueError('invalid network: theta %r of reticulation %d is not in [0, 1]' % (v.theta, v.idx))
    # level-1: the cycles (found by network_cycles, which refuses a reticulation on a cycle) are vertex-disjoint
    used = {}
    for h, top, s0, s1 in network_cycles(root):
        for x in [top] + s0 + s1 + [h]:
            if x in used:
                raise ValueError('invalid network: node %d lies on the cycles of reticulations %d and %d (level > 1)' % (x, used[x], h))
            used[x] = h
    return order


def is_tree_child(root):
    """every internal node has at least one child that is not a reticulation"""
    return all(v.kind == 'leaf' or any(c.kind != 'ret' for c in v.children) for v in all_nodes(root))


# ============================================================================ 3. cycles, undirected view, re-rooting
def rooted_topology_id(root, n_leaves):
    return network_canonical_string(root, n_leaves)


def _parent_map(root):
    parents = {}
    for v in all_nodes(root):
        for c in v.children:
            parents.setdefault(c.idx, []).append(v.idx)
    return parents


def network_cycles(root):
    """for every reticulation h: (h.idx, top_idx, side0, side1): side_k = node indices from h's slot-k parent up
    to, excluding, the top node of the cycle.  ValueError when another reticulation lies on the cycle or the two
    parent paths do not meet (not level-1)."""
    parents = _parent_map(root)

    def is_hybrid(idx):
        return len(parents.get(idx, [])) == 2
    out = []
    for h in reticulation_nodes(root):
        up0 = []
        v = h.parents[0].idx
        while True:                                   # climb from the slot-0 parent to the root or below a reticulation
            if is_hybrid(v):
                raise ValueError('invalid network: a reticulation lies directly above reticulation %d (not level-1)' % h.idx)
            up0.append(v)
            if v not in parents:
                break
            nxt = parents[v][0]
            if is_hybrid(nxt):
                break
            v = nxt
        pos = {x: i for i, x in enumerate(up0)}
        side1 = []
        v = h.parents[1].idx
        while v not in pos:                           # climb from the slot-1 parent until the slot-0 path is met
            if is_hybrid(v) or v not in parents:
                raise ValueError('invalid network: the parent paths of reticulation %d do not meet below a reticulation or the root (not level-1)' % h.idx)
            side1.append(v)
            v = parents[v][0]
        top = v
        out.append((h.idx, top, up0[:pos[top]], side1))
    return out


class SemiDirected(object):
    """The undirected multigraph of a rooted network with its degree-2 root suppressed, plus the hybrid-edge
    annotations: the "semi-directed" view.

    nodes   {idx: NetNode}                      (the root excluded)
    edges   {eid: {'u','v','length','hybrid','gamma'}}; 'hybrid' = idx of the reticulation the edge enters (else
            None), 'gamma' its inheritance probability.  The root's two edges are merged (lengths summed).
    adj     {idx: [(neighbour, eid)]}
    rings   [(h_idx, ring_nodes, ring_edges)] one per reticulation: its cycle in cyclic order, ring_edges[i] between
            ring_nodes[i] and ring_nodes[(i + 1) % k].  k = 2 (two parallel hybrid edges) happens exactly when the
            root is the top of a rooted 3-cycle.
    below   indices of the nodes strictly below a reticulation
    root_edges  eids on which the root can be placed so that the rooted network is again in the supported class:
            edges with no endpoint in `below`; when a ring has k = 2 only its two parallel edges (any other root
            position would produce parallel edges p => h, which the sampler cannot generate)."""

    def __init__(self, root):
        self.root = root
        nodes = {v.idx: v for v in all_nodes(root)}
        if len(root.children) != 2:
            raise ValueError('invalid network: root with %d children' % len(root.children))
        self.nodes = {i: v for i, v in nodes.items() if i != root.idx}
        self.edges, self.adj, dag_eid = {}, {i: [] for i in self.nodes}, {}
        seen_from = {}

        def gamma_of(h, parent):
            slots = [k for k in (0, 1) if h.parents[k] is parent]
            if not slots:
                raise ValueError('invalid network: node %d is not a parent of reticulation %d' % (parent.idx, h.idx))
            k = seen_from.get((parent.idx, h.idx), 0)
            seen_from[(parent.idx, h.idx)] = k + 1
            slot = slots[min(k, len(slots) - 1)]
            return h.theta if slot == 0 else 1.0 - h.theta

        def add_edge(u, v, length, hybrid=None, gamma=None):
            eid = len(self.edges)
            self.edges[eid] = {'u': u, 'v': v, 'length': float(length), 'hybrid': hybrid, 'gamma': gamma}
            self.adj[u].append((v, eid)); self.adj[v].append((u, eid))
            return eid
        for v in nodes.values():
            if v is root:
                continue
            for c, d in zip(v.children, v.child_dists):
                if c.kind == 'ret':
                    eid = add_edge(v.idx, c.idx, d, c.idx, gamma_of(c, v))
                else:
                    eid = add_edge(v.idx, c.idx, d)
                dag_eid.setdefault((v.idx, c.idx), []).append(eid)
        (a, da), (b, db) = zip(root.children, root.child_dists)
        hyb, gam = None, None
        for c in (a, b):
            if c.kind == 'ret':
                if hyb is not None:
                    raise ValueError('invalid network: both root children are reticulations')
                hyb, gam = c.idx, gamma_of(c, root)
        m = add_edge(a.idx, b.idx, da + db, hyb, gam)
        dag_eid.setdefault((root.idx, a.idx), []).append(m)
        dag_eid.setdefault((root.idx, b.idx), []).append(m)
        self.root_eid = m
        self.dag_eid = dag_eid
        # rings
        self.rings = []
        for h, top, s0, s1 in network_cycles(root):
            ring = [top] + list(reversed(s0)) + [h] + s1          # top -> slot-0 side -> h -> slot-1 side -> top
            closing = None
            if top == root.idx:
                ring = ring[1:]
                closing = m
            k = len(ring)
            ring_edges, used = [], set()
            for i in range(k):
                x, y = ring[i], ring[(i + 1) % k]
                if i == k - 1 and closing is not None:
                    ring_edges.append(closing)
                    continue
                cands = [e for e in dag_eid.get((x, y), []) + dag_eid.get((y, x), []) if e not in used]
                if not cands:
                    raise ValueError('invalid network: no edge between cycle nodes %d and %d' % (x, y))
                ring_edges.append(cands[0]); used.add(cands[0])
            self.rings.append((h, ring, ring_edges))
        self.ring_of_edge = {}
        for ci, (h, rn, re_) in enumerate(self.rings):
            for e in re_:
                self.ring_of_edge[e] = ci
        # below
        self.below = set()
        for h in reticulation_nodes(root):
            stack = list(h.children)
            while stack:
                v = stack.pop()
                if v.idx in self.below:
                    continue
                self.below.add(v.idx)
                stack.extend(v.children)
        digons = [re_ for (h, rn, re_) in self.rings if len(rn) == 2]
        if len(digons) > 1:
            raise ValueError('invalid network: two pairs of parallel hybrid edges')
        if digons:
            self.root_edges = list(digons[0])
        else:
            self.root_edges = [eid for eid, e in self.edges.items() if e['u'] not in self.below and e['v'] not in self.below]

    def leaves_from(self, start, blocked_edges):
        """leaf indices reachable from `start` without crossing the edges in `blocked_edges`"""
        seen, stack, out = {start}, [start], set()
        while stack:
            x = stack.pop()
            if self.nodes[x].kind == 'leaf':
                out.add(x)
            for y, eid in self.adj[x]:
                if eid in blocked_edges or y in seen:
                    continue
                seen.add(y); stack.append(y)
        return frozenset(out)


def reroot(root, eid_root, sd=None, split=0.5):
    """Rooted NetNode DAG of the same semi-directed network with the root on the undirected edge `eid_root` of
    SemiDirected(root); the edge length is divided split : 1 - split (the likelihood and the prior depend only on
    the sum).  The result is in the supported class iff eid_root is in sd.root_edges; other positions raise
    ValueError when a reticulation would be entered from below, or return a network with parallel edges."""
    sd = sd if sd is not None else SemiDirected(root)
    nodes, adj, edges = sd.nodes, sd.adj, sd.edges
    e = edges[eid_root]
    a, b = e['u'], e['v']
    R = max(max(nodes), root.idx) + 1
    rings = [(h, list(rn), list(re_)) for (h, rn, re_) in sd.rings]
    cycle_of, cycle_edges = {}, set()
    for ci, (h, rn, re_) in enumerate(rings):
        for x in rn:
            cycle_of[x] = ci
        cycle_edges.update(re_)
    children = {idx: [] for idx in adj}             # idx -> [(child idx, edge token)]; token = eid or ('root', child)
    children[R] = []
    visited, entered, queue = {R}, set(), deque()

    def attach(u, v, token):
        children[u].append((v, token))
        if v not in visited:
            visited.add(v)
            queue.append(v)
            if v in cycle_of and cycle_of[v] not in entered:
                enter_cycle(cycle_of[v], v)

    def enter_cycle(ci, entry):
        h, rn, re_ = rings[ci]
        entered.add(ci)
        if entry == h:
            raise ValueError('invalid root position: reticulation %d would be entered from below' % h)
        k = len(rn)
        i = rn.index(entry)
        x = i                                        # forward arc: rn[x] -> rn[x+1] through re_[x]
        while rn[x] != h:
            attach(rn[x], rn[(x + 1) % k], re_[x])
            x = (x + 1) % k
        x = i                                        # backward arc: rn[x] -> rn[x-1] through re_[x-1]
        while rn[x] != h:
            attach(rn[x], rn[(x - 1) % k], re_[(x - 1) % k])
            x = (x - 1) % k
    if eid_root in cycle_edges:
        ci = next(k for k, (h, rn, re_) in enumerate(rings) if eid_root in re_)
        h, rn, re_ = rings[ci]
        j = re_.index(eid_root)                      # the edge between rn[j] and rn[j+1]
        rn2 = rn[:j + 1] + [R] + rn[j + 1:]
        re2 = re_[:j] + [('root', rn[j]), ('root', rn[(j + 1) % len(rn)])] + re_[j + 1:]
        rings[ci] = (h, rn2, re2)
        cycle_of[R] = ci
        enter_cycle(ci, R)
    else:
        attach(R, a, ('root', a))
        attach(R, b, ('root', b))
    while queue:
        u = queue.popleft()
        for v, eid in adj[u]:
            if eid in cycle_edges or v in visited:
                continue                             # cycle edges are oriented by enter_cycle
            attach(u, v, eid)
    if len(visited) != len(adj) + 1:
        raise ValueError('invalid root position: not every node is reachable')
    new = {idx: NetNode(idx, v.kind) for idx, v in nodes.items()}
    for idx, v in nodes.items():
        new[idx].ret_id = v.ret_id
    new_root = NetNode(R, 'tree')
    new[R] = new_root

    def edge_data(token):
        if isinstance(token, tuple):
            frac = split if token[1] == a else 1.0 - split
            return e['length'] * frac, e['gamma'], e['hybrid']
        ed = edges[token]
        return ed['length'], ed['gamma'], ed['hybrid']
    for u, lst in children.items():
        for v, token in lst:
            d, g, hyb = edge_data(token)
            new[u].children.append(new[v])
            new[u].child_dists.append(d)
            if new[v].kind == 'ret':
                h = new[v]
                if g is None or hyb != v:
                    raise ValueError('invalid root position: edge (%d, %d) enters a reticulation but is not a hybrid edge' % (u, v))
                if h.parents[0] is None:
                    h.parents[0] = new[u]; h.parent_dists[0] = d; h.theta = g
                elif h.parents[1] is None:
                    h.parents[1] = new[u]; h.parent_dists[1] = d
                    if abs(h.theta + g - 1.0) > 1e-9:
                        raise ValueError('inconsistent inheritance probabilities at reticulation %d' % v)
                else:
                    raise ValueError('invalid root position: reticulation %d received three parents' % v)
    for idx, w in new.items():
        if w.kind == 'ret':
            if len(w.children) != 1 or w.parents[1] is None:
                raise ValueError('invalid root position: reticulation %d is not entered from above twice' % idx)
            w.child_dist = w.child_dists[0]
        elif w.kind == 'tree' and len(w.children) != 2:
            raise ValueError('invalid root position: tree node %d has %d children' % (idx, len(w.children)))
    assign_tokens(new_root)
    return new_root


def in_class_rootings(root, sd=None):
    """[(eid, rooted network)] for every root position of sd.root_edges (all are networks of the supported class)"""
    sd = sd if sd is not None else SemiDirected(root)
    return [(eid, reroot(root, eid, sd)) for eid in sd.root_edges]


def semi_directed_topology_id(root, n_leaves, return_details=False, sd=None):
    """Canonical id of the root-equivalence class of the network: the lexicographically smallest rooted canonical
    string over its in-class rootings.  Two networks of the supported class have the same id iff one is a
    re-rooting of the other.  With return_details: (id, number of in-class root positions, number of distinct rooted
    topologies among them)."""
    sd = sd if sd is not None else SemiDirected(root)
    strings = [network_canonical_string(r2, n_leaves) for _, r2 in in_class_rootings(root, sd)]
    if not strings:
        raise ValueError('no in-class root position')
    best = min(strings)
    return (best, len(strings), len(set(strings))) if return_details else best


def _edge_type(sd, eid):
    e = sd.edges[eid]
    if eid in sd.ring_of_edge:
        return 'cycle'
    if sd.nodes[e['u']].kind == 'leaf' or sd.nodes[e['v']].kind == 'leaf':
        return 'pendant'
    return 'internal'


def root_position_type(root, sd=None):
    """where the root of this rooted network sits in its semi-directed view: 'pendant' (on a leaf edge),
    'cycle' (on a cycle edge, hybrid edges included) or 'internal' (any other edge)"""
    sd = sd if sd is not None else SemiDirected(root)
    return _edge_type(sd, sd.root_eid)


def root_position_type_counts(root, sd=None):
    """number of in-class root positions of each type {'pendant', 'cycle', 'internal'}: under the target the root
    is uniformly distributed over the in-class root positions of a semi-directed network (root-position
    uniformity diagnostic)"""
    sd = sd if sd is not None else SemiDirected(root)
    out = {'pendant': 0, 'cycle': 0, 'internal': 0}
    for eid in sd.root_edges:
        out[_edge_type(sd, eid)] += 1
    return out


# ============================================================================ 4. displayed trees and RF
def leaf_sets_of_tree(ch, tree_root):
    sets, stack = {}, [(tree_root, False)]
    while stack:
        v, done = stack.pop()
        if v in sets:
            continue
        kids = [c for c, _ in ch.get(v, [])]
        if not kids:
            sets[v] = frozenset([v])
        elif done:
            sets[v] = frozenset().union(*[sets[c] for c in kids])
        else:
            stack.append((v, True))
            stack.extend((c, False) for c in kids)
    return sets


def unrooted_splits(ch, tree_root, n_leaves):
    """non-trivial splits of a displayed tree seen as an UNROOTED tree: for every edge the side not containing
    leaf 0, kept when both sides have at least two leaves"""
    sets = leaf_sets_of_tree(ch, tree_root)
    full = frozenset(range(n_leaves))
    splits = set()
    for v, lst in ch.items():
        for c, _ in lst:
            side = sets[c]
            if 0 in side:
                side = full - side
            if 2 <= len(side) <= n_leaves - 2:
                splits.add(side)
    return frozenset(splits)


def splits_id(splits):
    """canonical id of an unrooted tree topology: its sorted non-trivial splits written out"""
    return '|'.join(','.join(str(x) for x in sorted(s)) for s in sorted(splits, key=lambda s: (len(s), sorted(s))))


def rf_distance(splits_a, splits_b):
    """unrooted Robinson-Foulds distance = number of non-trivial splits present in exactly one tree (a metric on
    unrooted tree topologies; maximum 2 (n - 3) for binary trees on n >= 4 leaves)"""
    return len(splits_a ^ splits_b)


def rf_max(n_leaves):
    return max(1, 2 * (n_leaves - 3))


def displayed_tree_table(root, n_leaves):
    """one dict per displayed tree (2^R entries, duplicates of a topology kept): weight = product of the inheritance
    probabilities of the kept hybrid edges, splits, id, tree = (children dict, tree root)"""
    out = []
    for w, ch, tr in displayed_trees(root):
        sp = unrooted_splits(ch, tr, n_leaves)
        out.append({'weight': float(w), 'splits': sp, 'id': splits_id(sp), 'tree': (ch, tr)})
    return out


def displayed_topology_distribution(table):
    """{topology id: {'weight': summed weight, 'splits': split set}}: the probability, under the displayed-tree
    model, that a site follows a tree of that unrooted topology.  Weights sum to 1."""
    out = {}
    for d in table:
        e = out.setdefault(d['id'], {'weight': 0.0, 'splits': d['splits']})
        e['weight'] += d['weight']
    total = sum(e['weight'] for e in out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError('displayed-tree weights sum to %r, not 1' % total)
    return out


def major_topology(dist):
    """(id of the displayed-tree topology with the largest summed weight, tie flag); ties -> the smallest id"""
    best = max(e['weight'] for e in dist.values())
    ids = sorted(k for k, e in dist.items() if abs(e['weight'] - best) <= 1e-12)
    return ids[0], len(ids) > 1


def displayed_tree_newick(ch, tree_root, taxa):
    def rec(v, length):
        tail = '' if length is None else ':' + _fmt(length)
        if v not in ch or not ch[v]:
            return quote_label(taxa[v]) + tail
        return '(' + ','.join(rec(c, d) for c, d in ch[v]) + ')' + tail
    return rec(tree_root, None) + ';'


# ============================================================================ 5. reticulation (cycle) signatures
def reticulation_signatures(root, n_leaves, sd=None):
    """One record per reticulation, computed on the semi-directed view (hence the same for every in-class rooting):

    'hybrid'  sorted leaf indices below the reticulation (the hybrid block H)
    'cycle'   the pendant leaf blocks of the other cycle nodes in cyclic order starting next to H, in the direction
              that makes the sequence lexicographically smaller (tuple of sorted tuples); its first and last entries
              are the blocks through which the two hybrid edges leave H: the hybrid parent locations
    'gammas'  inheritance probabilities of the hybrid edges towards cycle[0] and cycle[-1]
              (for k_sd = 2, the two parallel edges: sorted decreasingly)
    'k_sd'    number of nodes of the cycle in the semi-directed view; k_sd <= 3 <=> the two displayed trees obtained
              by switching this reticulation have the same unrooted topology
    'k_rooted' number of nodes of the cycle in the rooted network; 'root_on_cycle' whether the root is one of them
    'key'     (hybrid, cycle): the hashable taxon-based signature of the reticulation placement."""
    sd = sd if sd is not None else SemiDirected(root)
    rooted = {h: (top, s0, s1) for h, top, s0, s1 in network_cycles(root)}
    out = []
    for h, rn, re_ in sd.rings:
        k = len(rn)
        blocked = set(re_)
        i = rn.index(h)
        H = tuple(sorted(sd.leaves_from(h, blocked)))
        fwd = [tuple(sorted(sd.leaves_from(rn[(i + t) % k], blocked))) for t in range(1, k)]
        if k == 2:
            g = sorted((sd.edges[re_[0]]['gamma'], sd.edges[re_[1]]['gamma']), reverse=True)
            cyc, gam = tuple(fwd), tuple(g)
        else:
            g_fwd = sd.edges[re_[i]]['gamma']                    # edge between h and rn[i+1]
            g_bwd = sd.edges[re_[(i - 1) % k]]['gamma']          # edge between rn[i-1] and h
            bwd = fwd[::-1]
            if tuple(bwd) < tuple(fwd):
                cyc, gam = tuple(bwd), (g_bwd, g_fwd)
            else:
                cyc, gam = tuple(fwd), (g_fwd, g_bwd)
        top, s0, s1 = rooted[h]
        out.append({'hybrid': H, 'cycle': cyc, 'gammas': (float(gam[0]), float(gam[1])), 'k_sd': k,
                    'k_rooted': 2 + len(s0) + len(s1), 'root_on_cycle': top == root.idx, 'key': (H, cyc)})
    out.sort(key=lambda s: s['key'])
    return out


def signatures_to_json(sigs, taxa):
    """lossless JSON text of the signatures with taxon names (lists of lists; no delimiter-joined strings)"""
    return json.dumps([{'hybrid': [taxa[i] for i in s['hybrid']], 'cycle': [[taxa[i] for i in b] for b in s['cycle']],
                        'gammas': list(s['gammas']), 'k_sd': s['k_sd'], 'k_rooted': s['k_rooted'],
                        'root_on_cycle': bool(s['root_on_cycle'])} for s in sigs], ensure_ascii=False)


# ============================================================================ 6. root-invariant parameter keys
def semi_directed_edge_table(root, n_leaves, sd=None):
    """(lengths, gammas): two dictionaries with taxon-based keys that are identical for every in-class rooting and
    every node numbering, child order and slot labelling of the same semi-directed network.

    lengths: one entry per edge of the semi-directed view (the two root edges count as one edge of length
             b_left + b_right):
               ('cut', side, below_hybrid)   a non-cycle edge: `side` = leaf set on the side without leaf 0;
                                             below_hybrid marks the edge from a reticulation to its child
               ('cyc', H, {block_u, block_v}) a cycle edge between the cycle nodes with pendant blocks block_u, block_v
               ('cyc2', H, rank)             the two parallel hybrid edges of a 2-cycle, rank 0 = larger gamma
    gammas:  ('gamma', H, block_of_parent) -> inheritance probability of the hybrid edge from that cycle node;
             ('gamma2', H, rank) for a 2-cycle.
    ValueError if two edges receive the same key (guarded, not assumed)."""
    sd = sd if sd is not None else SemiDirected(root)
    lengths, gammas = {}, {}

    def put(table, key, value):
        if key in table:
            raise ValueError('edge key collision: %r' % (key,))
        table[key] = float(value)
    for h, rn, re_ in sd.rings:
        blocked = set(re_)
        k = len(rn)
        blocks = [sd.leaves_from(x, blocked) for x in rn]
        H = blocks[rn.index(h)]
        if k == 2:
            order = sorted(range(2), key=lambda t: -sd.edges[re_[t]]['gamma'])
            for rank, t in enumerate(order):
                put(lengths, ('cyc2', H, rank), sd.edges[re_[t]]['length'])
                put(gammas, ('gamma2', H, rank), sd.edges[re_[t]]['gamma'])
            continue
        for i in range(k):
            ed = sd.edges[re_[i]]
            pair = frozenset((blocks[i], blocks[(i + 1) % k]))
            put(lengths, ('cyc', H, pair), ed['length'])
            if ed['hybrid'] is not None:
                other = blocks[(i + 1) % k] if rn[i] == h else blocks[i]
                put(gammas, ('gamma', H, other), ed['gamma'])
    for eid, ed in sd.edges.items():
        if eid in sd.ring_of_edge:
            continue
        side = sd.leaves_from(ed['u'], {eid})
        if 0 in side:
            side = frozenset(range(n_leaves)) - side
        below_hybrid = sd.nodes[ed['u']].kind == 'ret' or sd.nodes[ed['v']].kind == 'ret'
        put(lengths, ('cut', side, below_hybrid), ed['length'])
    if len(lengths) != len(sd.edges):
        raise ValueError('edge table has %d keys for %d edges' % (len(lengths), len(sd.edges)))
    return lengths, gammas


# ============================================================================ 7. mu-representation
def mu_vectors(root, n_leaves):
    """Counter of the path-count vectors mu(u) = (number of directed paths from u to leaf i)_i over all nodes u:
    the mu-representation of Cardona, Rossello & Valiente (2009).  The symmetric difference of two such multisets
    is the mu-distance, a metric on tree-child networks with the same leaves; for trees it is the rooted
    Robinson-Foulds (cluster) distance."""
    order = _topological_order(root)
    mu = {}
    for v in reversed(order):
        if v.kind == 'leaf':
            vec = [0] * n_leaves
            vec[v.idx] = 1
            mu[v.idx] = tuple(vec)
        else:
            mu[v.idx] = tuple(sum(col) for col in zip(*[mu[c.idx] for c in v.children]))
    return Counter(mu.values())


def mu_distance(mu_a, mu_b):
    """size of the symmetric difference of two mu-representations (multisets)"""
    return sum(abs(mu_a.get(k, 0) - mu_b.get(k, 0)) for k in set(mu_a) | set(mu_b))


# ============================================================================ structural summaries
def count_nodes_edges(root):
    nodes = all_nodes(root)
    return len(nodes), sum(len(v.children) for v in nodes)


def branch_lengths(root):
    """(all rooted branch lengths parent -> child, [(ret_id, theta)])"""
    L = [float(d) for v in all_nodes(root) for d in v.child_dists]
    return L, [(h.ret_id, float(h.theta)) for h in reticulation_nodes(root)]


def rooted_edge_table(root, n_leaves):
    """(lengths, gammas) keyed by (canonical string of the parent, canonical string of the child): identifies every
    edge of a network of the supported class independently of node numbers, child order and slot labels (distinct
    nodes have distinct canonical strings there; a collision raises ValueError).  Used to compare two serialisations
    of the SAME rooted network edge by edge; gammas holds the hybrid edges only."""
    order = _topological_order(root)
    canon = {}
    for v in reversed(order):
        if v.kind == 'leaf':
            canon[v.idx] = 'L%d' % v.idx
        elif v.kind == 'ret':
            canon[v.idx] = '#' + canon[v.children[0].idx]
        else:
            canon[v.idx] = '(' + ','.join(sorted(canon[c.idx] for c in v.children)) + ')'
    lengths, gammas = {}, {}
    for v in order:
        for c, d in zip(v.children, v.child_dists):
            key = (canon[v.idx], canon[c.idx])
            if key in lengths:
                raise ValueError('rooted edge key collision at edge (%d, %d)' % (v.idx, c.idx))
            lengths[key] = float(d)
            if c.kind == 'ret':
                gammas[key] = float(c.theta) if c.parents[0] is v else 1.0 - float(c.theta)
    return lengths, gammas
