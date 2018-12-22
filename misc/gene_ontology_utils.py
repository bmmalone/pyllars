""" Utilities for working with the Gene Ontology (GO)

Please see the GO website for more details (http://www.geneontology.org/).

These utilities are often designed to work with `goatools`.

* paper: https://www.nature.com/articles/s41598-018-28948-z
* git repository: https://github.com/tanghaibao/goatools
"""

import collections

import goatools.associations

###
# Helpful constants and data structures
###

HUMAN_NCBI_TAXON_ID = 9606

BIOLOGICAL_PROCESS_ROOT_TERM = 'GO:0008150'
CELLULAR_COMPONENT_ROOT_TERM = 'GO:0005575'
MOLECULAR_FUNCTION_ROOT_TERM = 'GO:0003674'

go_sub_ontology_terms = [
    'biological_processes',
    'cellular_components',
    'molecular_functions'
]

go_sub_ontology_terms = collections.namedtuple(
    'go_sub_ontology_terms',
    ",".join(go_sub_ontology_terms)
)

###
# I looked these up here:
#    https://www.ebi.ac.uk/QuickGO/term/GO:0001011
###

_DEPRECATED_SYNONYMS = {
    'GO:0001077': 'GO:0001228',
    'GO:0003705': 'GO:0000981',
    'GO:0001078': 'GO:0001227',
    'GO:1900390': 'GO:0034756',
    'GO:0001205': 'GO:0001228',
    'GO:0001206': 'GO:0001227',
    'GO:1903758': 'GO:0097549',
    'GO:0001075': 'GO:0016251',
    'GO:0000982': 'GO:0000981',
    'GO:0097286': 'GO:0006826',
    'GO:0000983': 'GO:0016251',
    'GO:1903874': 'GO:0034755',
    'GO:0015684': 'GO:0006826',
    'GO:0070627': 'GO:0033212',
    'GO:0001011': 'GO:0140223',
    'GO:0005575': 'GO:0008372',
    'GO:0008150': 'GO:0008150',
    'GO:0003674': 'GO:0003674'
}

_DEPRECATED_TERM_SUBONTOLOGIES = {
    'GO:0001077': 'molecular_function',
    'GO:0003705': 'molecular_function',
    'GO:0001078': 'molecular_function',
    'GO:1900390': 'biological_process',
    'GO:0001205': 'molecular_function',
    'GO:0001206': 'molecular_function',
    'GO:1903758': 'biological_process',
    'GO:0001075': 'molecular_function',
    'GO:0000982': 'molecular_function',
    'GO:0097286': 'biological_process',
    'GO:0000983': 'molecular_function',
    'GO:1903874': 'biological_process',
    'GO:0015684': 'biological_process',
    'GO:0070627': 'biological_process',
    'GO:0001011': 'molecular_function',
    'GO:0005575': 'cellular_component',
    'GO:0008150': 'biological_process',
    'GO:0003674': 'molecular_function'
}

_DEPRECATED_SUBONTOLOGY_TERM_SETS = {
    'molecular_function': {
        k for k,v in _DEPRECATED_TERM_SUBONTOLOGIES.items() if v == 'molecular_function'
    },
    'biological_process': {
        k for k,v in _DEPRECATED_TERM_SUBONTOLOGIES.items() if v == 'biological_process'
    },
    'cellular_component': {
        k for k,v in _DEPRECATED_TERM_SUBONTOLOGIES.items() if v == 'cellular_component'
    },
}

###
# File IO helpers
###

def load_human_gene2go_annotations(gene2go):
    """ Load the GO annotations for human from the gene2go file
    
    Parameters
    ----------
    gene2go : path-like
        The path to the gene2go file
        
    Returns
    -------
    human_gene2go : defaultdict of a string to a set of strings
        A map from the human Entrez gene IDs to the associated GO annotations
    
    """
    
    gene2go_human = goatools.associations.read_ncbi_gene2go(
        gene2go, taxids=[HUMAN_NCBI_TAXON_ID]
    )
    
    return gene2go_human

###
# Helpers for searching through the GO directed acyclic graph
###

def get_ancestor_terms(go_term, go_dag, include_go_term=True):
    """ Find all ancestors of the given term
    
    Optionally, based on `include_go_term`, `go_term` itself will be
    included in the returned set.
    
    Parameters
    ----------
    go_term : string
        The GO term. Example: "GO:0008150"
        
    go_dag : goatools.obo_parser.GODag
        The Gene Ontology DAG, parsed from the source obo file
        
    include_go_term : bool
        Whether to incldue `go_term` itself in the set of ancestors
        
    Returns
    -------
    ancestors : set
        The set of ancestors for this term. If `include_go_term` is True, then
        this set will also include `go_term`.
    """
    paths = go_dag.paths_to_top(go_term)
    
    all_ancestors = {go_term}
    if paths is not None:
        all_ancestors = {
            term.id for path in paths for term in path
        }

    if not include_go_term:
        all_ancestors.remove(go_term)
        
    return all_ancestors

def _dfs_set(term, ontology_terms, max_depth=-1):
    """ Recursively add all descendents of `term` to `ontology_terms`
    
    This helper function expands one node in a depth-first search of the
    Gene Ontology DAG. It builds up a *set* of all descendents. It is not
    intended for external use.
    """
    
    if term.depth == max_depth:
        return
    
    for child in term.children:
        ontology_terms.add(child.id)        
        _dfs_set(child, ontology_terms)

def get_descendent_terms(go_term, go_dag, include_go_term=True):
    """ Search through the GO DAG to find all descendent terms of `go_term`.
    
    For example, if `go_term` is one of the "main" sub-ontology roots, then this
    function returns all GO terms from the given sub-ontology.
    
    Parameters
    ----------
    go_term : string
        The GO term from which the search will begin. Example: "GO:0008150"
        
    go_dag : goatools.obo_parser.GODag or None
        The Gene Ontology DAG, parsed from the source obo file. 
        
    include_go_term : bool
        Whether to incldue `go_term` itself in the list of descendents
        
    Returns
    -------
    descendents : set of strings
        A set containing all terms which are descendents of `go_term`. If
        `include_go_term` is True, then this set will also include `go_term`.
    """
    
    descendents = set()
    goatools_term = go_dag[go_term]
    _dfs_set(goatools_term, descendents)

    if not include_go_term:
        descendents.remove(go_term)
    
    return descendents

def get_sub_ontology_terms(go_dag):
    """ Extract the terms from each sub-ontology into a set
    
    Parameters
    ----------
    go_dag : goatools.obo_parser.GODag or None
        The Gene Ontology DAG, parsed from the source obo file. 
        
    Returns
    -------
    go_sub_ontology_terms : namedtuple
        A named tuple with the following fields:
        * biological_processes
        * cellular_components
        * molecular_functions
        
        Each field is a set containing terms from the respective sub-ontology.
    """
    biological_process_terms = get_descendent_terms(BIOLOGICAL_PROCESS_ROOT_TERM, go_dag)
    cellular_component_terms = get_descendent_terms(CELLULAR_COMPONENT_ROOT_TERM, go_dag)
    molecular_function_terms = get_descendent_terms(MOLECULAR_FUNCTION_ROOT_TERM, go_dag)
    
    # also, add the deprecated terms
    biological_process_terms |= _DEPRECATED_SUBONTOLOGY_TERM_SETS['biological_process']
    cellular_component_terms |= _DEPRECATED_SUBONTOLOGY_TERM_SETS['cellular_component']
    molecular_function_terms |= _DEPRECATED_SUBONTOLOGY_TERM_SETS['molecular_function']
    
    ret = go_sub_ontology_terms(
        biological_process_terms,
        cellular_component_terms,
        molecular_function_terms
    )
    
    return ret

###
# Helpers for working with gene-GO annotation mappings
###

def get_gene_go_annotations(gene, gene2go, include_ancestors=False, go_dag=None):
    """ Retrieve the GO terms associated with `gene`
    
    Parameters
    ----------
    gene : string
        The identifier for the gene. Presumably, this should be the Entrez gene
        identifier. Example: "5133" is the identifier for PDCD1, programmed cell
        death 1.
        
    gene2go : dictionary mapping gene identifiers to GO annotations
        The GO annotations for all genes of interest. Presumably, this comes from
        the gene2go annotations distributed by NCBI.
        
    include_ancestors : bool
        Whether to include just the direct annotations for `gene` (False) or
        all annotated terms as well as all of their ancestor terms in the GO
        hierarchy (given by `go_dag`)
        
    go_dag : goatools.obo_parser.GODag or None
        The Gene Ontology DAG, parsed from the source obo file. This is only
        necessary if `include_ancestors` is True.
        
    Returns
    -------
    annotations : list of pairs of (`gene`, `go_term`)
        The GO annotations for `gene`. If `include_ancestors` is True, then this
        list will include all annotated GO terms as well as all ancestors. Otherwise,
        it just includes the direct annotations.
    """
    
    if include_ancestors:
        if go_dag is None:
            msg = "Must have obodag to find ancestors"
            raise ValueError(msg)
            
        annotations = [
            (gene, ancestor)
                for go_term in gene2go[gene]
                    for ancestor in get_ancestor_terms(go_term, go_dag)
        ]
        
    else:
        annotations = [
            (gene, go_term) for go_term in gene2go[gene]
        ]
    return annotations