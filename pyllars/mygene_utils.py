"""
This module provides helper functions for querying the mygene.info
service via MyGene.py. Please consult the official documentation for
more details: http://docs.mygene.info/projects/mygene-py/en/latest/.
"""

import logging
logger = logging.getLogger(__name__)

import functools
import pandas as pd

import mygene

import pyllars.collection_utils as collection_utils
import pyllars.pandas_utils as pd_utils
import pyllars.utils as utils

import typing
from typing import Dict, Iterable, Optional

MYGENE_COLUMN_MAP = {
    'query': 'gene_id',
    'uniprot.Swiss-Prot': 'swiss_prot',
    'uniprot.TrEMBL': 'trembl'
}

MYGENE_COLUMNS = ['gene_id',
                  'symbol',
                  'swiss_prot',
                  'trembl',
                  'name', 
                  'summary',
                  'pdb',
                  'pfam',
                  'prosite'
                 ]

FIELDS = ("name,symbol,entrezgene,pdb,pfam,prosite,summary,uniprot.Swiss-Prot,"
          "uniprot.TrEMBL,go,interpro,pathway")

def get_entrez_gene_mapping(ensembl_protein_ids:Iterable[str]) -> Dict:
    """ Retrieve the Entrez gene ids for the given Ensembl protein identifiers
    
    Parameters
    ----------
    ensemble_protein_ids : typing.Iterable[str]
        The protein identifiers
        
    Returns
    -------
    mapping : typing.Dict
        A mapping from the Ensembl protein id to the Entrez id
    """
    mg = mygene.MyGeneInfo()

    records = mg.querymany(
        ensembl_protein_ids,
        scopes='ensembl.protein',
        fields='entrezgene',
        species='human'
    )
    
    mapping = {
        record['query']: record.get('entrezgene')
            for record in records
    }
    
    return mapping


def parse_go_terms(
        row:pd.Series,
        go_hierarchies:Iterable[str]=['BP', 'MF', 'CC']) -> Dict:
    """ Parse the GO terms out of the `go` field of `row`
    
    This function assumes the "go" field, if it is present, contains
    a dictionary in which the keys are the high-level GO hierarchies, and
    the values are lists of dictionaries which include a key "term" that
    gives the text description of interest.
    
    Parameters
    ----------
    row : pandas.Series
        Presumably, the result of a call to `mygene.get_genes` or similar
        
    go_hierarchies : typing.Iterable[str]                
        The top-level GO terms
            
    Returns
    -------
    go_terms : typing.Dict
        A dictionary containing the keys:
        * 'gene_id': which is retrieved from the 'query' field of row
        * 'go_terms': a semi-colon delimited list of the GO terms
    """

    go_terms = []
    
    if 'go' not in row:
        return None
    
    go = row['go']

    if isinstance(go, dict):
        for go_hierarchy in go_hierarchies:
            if go_hierarchy not in go:
                continue

            gh = utils.wrap_in_list(go[go_hierarchy])
            for go_term in gh:
                go_terms.append(go_term['term'])
    
    return {
        'gene_id': row['query'],
        'go_terms': ';'.join(go_terms)
    }

def parse_kegg_pathways(row:pd.Series) -> Dict:
    """ Parse all of the KEGG pathways out of the `kegg` field of `row`
    
    It first checks the `pathway` field, which is assumed to be a dictionary,
    of the row for a field key called `kegg`. It assumes the `kegg` field,
    if it is present, contains a list of dictionaries, and each dictionary
    contains a key called `name` which gives the name of that KEGG pathway.
        
    Parameters
    ----------
    row : pandas.Series
        Presumably, the result of a call to `mygene.get_genes` or similar
        
    Returns
    -------
    kegg_pathways : typing.Dict
        A dictionary containing the keys:
        * 'gene_id': which is retrieved from the 'query' field of row
        * 'kegg_pathways': a semi-colon delimited list of the KEGG pathways        
    """

    kps = []
    
    if 'pathway' not in row:
        return None
    
    p = row['pathway']

    if isinstance(p, dict):
        if 'kegg' in p:
            kegg_pathways = utils.wrap_in_list(p['kegg'])
            for kp in kegg_pathways:
                kps.append(kp['name'])
                
    return {
        'gene_id': row['query'],
        'kegg_pathways': ';'.join(kps)
    }

def parse_interpro(row:pd.Series) -> Dict:
    """ Parse the Interpro families out of the `interpro` field of `row`
    
    It assumes the `interpro` field, if it is present, contains a list of
    dictionaries, and each dictionary contains a key called `desc` which
    gives the description of that family.
         
    Parameters
    ----------
    row : pandas.Series
        Presumably, the result of a call to `mygene.get_genes` or similar
        
    Returns
    -------
    kegg_pathways : typing.Dict
        A dictionary containing the keys:
        * 'gene_id': which is retrieved from the 'query' field of row
        * 'interpro_families': a semi-colon delimited list of the Interpro families
    """
    interpro_terms = []
    
    if 'interpro' not in row:
        return None
    
    ip = row['interpro']
    
    if isinstance(ip, list):
        for ip_term in ip:
            interpro_terms.append(ip_term['desc'])
            
    return {
        'gene_id': row['query'],
        'interpro_families': ';'.join(interpro_terms)
    }

def query_mygene(
        gene_ids:Iterable[str],
        species:Optional[str]='all',
        scopes:Optional[str]=None,
        unique_only:bool=True, 
        mygene_url:str="http://mygene.info/v3") -> pd.DataFrame:
    """ Query mygene.info to find information about the `gene_ids`
    
    In particular, this function looks for information (via mygene) from
    Swiss-Prot, TrEMBL, Interpro, PDB, Pfam, PROSITE, the Gene Ontology,
    and KEGG.
        
    Parameters
    ----------
    gene_ids : typing.Iterable[str]
        A list of gene identifiers for the query. This list is passed
        directly to MyGeneInfo.getgenes, so any identifiers which are valid
        for it are valid here.

        As of 1.7.2016, it supports entrez/ensembl gene ids, where the entrez
        gene id can be either a string or integer.

        Example valid identifiers:
            ENSG00000004059 (from human)
            ENSMUSG00000051951 (from mouse)
            WBGene00022277 (from c. elegans)

    species : typing.Optional[str]
        Optionally, a comma-delimited list of species can be given. By default,
        all species are considered. For ensembl identifiers, this is unlikely
        to be a problem; for other scopes (such as gene symbols), it could cause
        unexpected behavior, though.

        Please see the mygene.info docs for more details about valid
        species: http://mygene.info/doc/data.html#species
    
    scopes : typing.Optional[str]
        Optionally, a comma-delimited list of scopes can be given for the
        `gene_ids` (so they need not necessarily be entrez/ensembl gene
        identifiers. 

        Please see the mygene.info docs for more details about 
        valid scopes: http://mygene.info/doc/query_service.html#available-fields

    unique_only : bool
        Whether to use only unique `gene_ids`. Typically, duplicates in the
        `gene_ids` list can cause problems for the joins in this function and
        cause huge memory usage. Unless some specific use-case is known,
        duplicates should be removed.

    mygene_url : str
        The url to use to connect to the mygene server. In principle, "v2" could
        be used instead of "v3".

    Returns
    -------
    df_gene_info : pandas.DataFrame
        A data frame with the following columns. All values are strings and separated
        by semi-colons if multiple values are present:

        * 'gene_id': the original query
        * 'swiss_prot': any matching Swiss-Prot identifier (e.g., "Q9N4D9")
        * 'trembl': any matching TrEMBL identifiers (e.g., "H2KZI1;Q95Y41")
        * 'name': a short description of the gene
        * 'summary': a longer description of the gene
        * 'pdb': any matching Protein Data Bank identifiers (e.g., "2B6H")
        * 'pfam': any matching Pfam protein families (e.g., "PF00254;PF00515")
        * 'prosite': any matching PROSITE protein domains, etc. (e.g., "PS50114;PS51156")
        * 'kegg_pathways': the name of any matching KEGG pathways
        * 'go_terms': the term name of any associated GO terms
        * 'interpro_families': the description of any associated Interpro family
    """

    MG = mygene.MyGeneInfo()
    MG.url = mygene_url
    
    # pull the information from mygene
    msg = "Pulling annotations from mygene.info"
    logger.info(msg)

    if unique_only:
        gene_ids = set(gene_ids)

    res = MG.getgenes(gene_ids, fields=FIELDS, as_dataframe=True, 
        species=species, scopes=scopes)
    res = res.reset_index()
    
    # parse out the various fields

    msg = "Parsing KEGG pathways"
    logger.info(msg)

    kps = pd_utils.apply(res, parse_kegg_pathways)
    kps = collection_utils.remove_nones(kps)
    kps_df = pd.DataFrame(kps)
    
    msg = "Parsing GO annotations"
    logger.info(msg)

    gos = pd_utils.apply(res, parse_go_terms)
    gos = collection_utils.remove_nones(gos)
    gos_df = pd.DataFrame(gos)
    
    msg = "Parsing Interpro families"
    logger.info(msg)

    interpros = pd_utils.apply(res, parse_interpro)
    interpros = collection_utils.remove_nones(interpors)
    interpros_df = pd.DataFrame(interpros)
    
    # make sure our columns of interest are there and have the right names
    msg = "Cleaning mygene data frame"
    logger.info(msg)

    res = res.rename(columns=MYGENE_COLUMN_MAP)
    
    for col in MYGENE_COLUMN_MAP.keys():
        if col not in res:
            res[col] = None
            
    for col in MYGENE_COLUMNS:
        if col not in res:
            res[col] = None
    
    # keep only the columns we care about
    res_df = res[MYGENE_COLUMNS]
    
    # merge the non-empty data frames
    msg = "Merging results"
    logger.info(msg)

    dfs = [res_df, kps_df, gos_df, interpros_df]
    dfs = [df for df in dfs if len(df.columns) > 0]
    
    res_df = pd_utils.join_df_list(dfs, join_col='gene_id', how='inner')
    
    # clean up the data frame and return it
    msg = "Cleaning mygene data frame, phase 2"
    logger.info(msg)

    res_df = res_df.drop_duplicates(subset='gene_id')
    res_df = res_df.fillna('')
    
    # fix the string vs. list of string fields
    sep_join = lambda l: ";".join(l)

    res_df['trembl'] = res_df['trembl'].apply(collection_utils.wrap_string_in_list)
    res_df['trembl'] = res_df['trembl'].apply(sep_join)

    res_df['pdb'] = res_df['pdb'].apply(collection_utils.wrap_string_in_list)
    res_df['pdb'] = res_df['pdb'].apply(sep_join)

    res_df['pfam'] = res_df['pfam'].apply(collection_utils.wrap_string_in_list)
    res_df['pfam'] = res_df['pfam'].apply(sep_join)

    res_df['prosite'] = res_df['prosite'].apply(collection_utils.wrap_string_in_list)
    res_df['prosite'] = res_df['prosite'].apply(sep_join)
    
    msg = "Finished compiling mygene annotations"
    logger.info(msg)

    return res_df