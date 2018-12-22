###
#   This module provides helper functions (namely, "query_mygene") for pulling
#   information from the mygene.info database.
###

import logging
logger = logging.getLogger(__name__)

import mygene

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

def get_entrez_gene_mapping(ensembl_protein_ids):
    """ Retrieve the Entrez gene ids for the given Ensembl protein identifiers
    
    Parameters
    ----------
    ensemble_protein_ids : iterable of strings
        The protein identifiers
        
    Returns
    -------
    mapping : dictionary
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


def parse_go_terms(row, go_hierarchies = ['BP', 'MF', 'CC']):
    """ This function parses all of the gene ontology terms out of the "go"
        field of a record. It assumes the "go" field, if it is present, contains
        a dictionary in which the keys are the high-level GO hierarchies, and
        the values are lists of dictionaries which include a key "term" that
        gives the text description of interest.
        
        Args:
            row (pd.Series, or dictionary): presumably, the result of a call
                to mygene.get_genes
                
            go_hierarchies (list of strings): the top-level GO terms
            
        Returns:
            dictionary: containing the keys:
                'gene_id': which is retrieved from the 'query' field of row
                'go_terms': a semi-colon delimited list of the GO terms

        Imports:
            misc.utils

    """
    import misc.utils as utils

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

def parse_kegg_pathways(row):
    """ This function parses all of the KEGG pathways out of the "kegg"
        field of a record. It first checks the "pathway" field, which is
        assumed to be a dictionary, of the row for a field key called 
        "kegg". It assumes the "kegg" field, if it is present,
        contains a list of dictionaries, and each dictionary contains a
        key called "name" which gives the name of that KEGG pathway.
        
        Args:
            row (pd.Series, or dictionary): presumably, the result of a call
                to mygene.get_genes
            
        Returns:
            dictionary: containing the keys:
                'gene_id': which is retrieved from the 'query' field of row
                'kegg_pathways': a semi-colon delimited list of the KEGG pathways

        Imports:
            misc.utils
    """
    import misc.utils as utils

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

def parse_interpro(row):
    """ This function parses all of the Interpro families out of the "interpro"
        field of a record. It assumes the "interpro" field, if it is present,
        contains a list of dictionaries, and each dictionary contains a key
        called "desc" which gives the description of that family.
        
        Args:
            row (pd.Series, or dictionary): presumably, the result of a call
                to mygene.get_genes
            
        Returns:
            dictionary: containing the keys:
                'gene_id': which is retrieved from the 'query' field of row
                'interpro_families': a semi-colon delimited list of the Interpro families
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

def query_mygene(gene_ids, species='all', scopes=None, unique_only=True, 
            mygene_url="http://mygene.info/v3"):

    """ This function uses mygene.info to find information about the
        genes specified in the provided list. In particular, this function
        looks for information (via mygene) from Swiss-Prot, TrEMBL,
        Interpro, PDB, Pfam, PROSITE, the Gene Ontology, and KEGG.
        
        Args:
            gene_ids (list-like): A list of gene identifiers for the query.
                This list is passed directly to MyGeneInfo.getgenes, so any
                identifiers which are valid for it are valid here.
                
                As of 1.7.2016, it supports entrez/ensembl gene ids, 
                where the entrez gene id can be either a string or integer.
                
                Example valid identifiers:
                    ENSG00000004059 (from human)
                    ENSMUSG00000051951 (from mouse)
                    WBGene00022277 (from c. elegans)

            scopes (csv string): Optionally, scopes can be given for the
                "gene_ids" (so they may not necessarily be entrez/ensembl
                gene identifiers. 
                
                Please see the mygene.info docs for more details about 
                valid scopes: 
                
                http://mygene.info/doc/query_service.html#available_fields
            
            species (csv string): The list of species to check. By default,
                all species are considered. For ensembl identifiers, this
                is unlikely to be a problem; for other scopes (such as gene
                symbols), it could cause unexpected behavior, though.

                Please see the mygene.info docs for more details about valid
                species:

                http://mygene.info/doc/data.html#species

            unique_only (bool): whether to use only unique gene_ids. Typically,
                duplicates in the gene_ids list can cause problems for the joins
                in this function and cause huge memory usage. Unless some 
                specific use-case is known, duplicates should be removed.

            mygene_url (string): the url to use to connect to the mygene server.
                In principle, "v2" could be used instead of "v3".
                    
        Returns:
            pd.DataFrame: a data frame with the following columns. All values are strings
                and separated by semi-colons if multiple values are present:
                
            'gene_id': the original query
            'swiss_prot': any matching Swiss-Prot identifier (e.g., "Q9N4D9")
            'trembl': any matching TrEMBL identifiers (e.g., "H2KZI1;Q95Y41")
            'name': a short description of the gene
            'summary': a longer description of the gene
            'pdb': any matching Protein Data Bank identifiers (e.g., "2B6H")
            'pfam': any matching Pfam protein families (e.g., "PF00254;PF00515")
            'prosite': any matching PROSITE protein domains, etc. (e.g., "PS50114;PS51156")
            'kegg_pathways': the name of any matching KEGG pathways
            'go_terms': the term name of any associated GO terms
            'interpro_families': the description of any associated Interpro family
                   

        Imports:
            functools
            pandas
            mygene
            misc.parallel
            misc.utils 
    """
    import functools
    import pandas as pd
    import mygene
    import misc.pandas_utils as pd_utils
    import misc.parallel as parallel
    import misc.utils as utils

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

    kps = parallel.apply_df_simple(res, parse_kegg_pathways)
    kps = [k for k in kps if k is not None]
    kps_df = pd.DataFrame(kps)
    
    msg = "Parsing GO annotations"
    logger.info(msg)

    gos = parallel.apply_df_simple(res, parse_go_terms)
    gos = [g for g in gos if g is not None]
    gos_df = pd.DataFrame(gos)
    
    msg = "Parsing Interpro families"
    logger.info(msg)

    interpros = parallel.apply_df_simple(res, parse_interpro)
    interpros = [i for i in interpros if i is not None]
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
    
    res_df = functools.reduce(lambda df1, df2: pd.merge(df1, df2, how='inner', on='gene_id'), dfs)
    
    # clean up the data frame and return it
    msg = "Cleaning mygene data frame, phase 2"
    logger.info(msg)

    res_df = res_df.drop_duplicates(subset='gene_id')
    res_df = res_df.fillna('')
    
    # fix the string vs. list of string fields
    sep_join = lambda l: ";".join(l)

    res_df['trembl'] = res_df['trembl'].apply(utils.wrap_string_in_list)
    res_df['trembl'] = res_df['trembl'].apply(sep_join)

    res_df['pdb'] = res_df['pdb'].apply(utils.wrap_string_in_list)
    res_df['pdb'] = res_df['pdb'].apply(sep_join)

    res_df['pfam'] = res_df['pfam'].apply(utils.wrap_string_in_list)
    res_df['pfam'] = res_df['pfam'].apply(sep_join)

    res_df['prosite'] = res_df['prosite'].apply(utils.wrap_string_in_list)
    res_df['prosite'] = res_df['prosite'].apply(sep_join)
    
    msg = "Finished compiling mygene annotations"
    logger.info(msg)

    return res_df


###
#   The following set of functions is used to map Ensembl transcript identifers
#   to Ensembl gene identifers.
###
def _parse_transcript_to_gene_query_result(entry):
    gene = None
    transcript = entry['query']
    ret = []
    
    if 'ensembl.gene' in entry:
        gene = entry['ensembl.gene']
        r = {
            'transcript_id': transcript,
            'gene_id': gene
        }
        ret.append(r)
        
    elif 'ensembl' in entry:
        for ensembl_entry in entry['ensembl']:
            if 'gene' in ensembl_entry:
                gene = ensembl_entry['gene']
                r = {
                    'transcript_id': transcript,
                    'gene_id': gene
                }
                ret.append(r)
                     
    return ret

def get_transcript_to_gene_mapping(transcript_ids):
    """ This function uses mygene.info to find the Ensembl gene identifers which
        match to the provided list of Ensembl transcript identifers. The result
        is returned as a pandas data frame which is useful for joining data
        frames with gene information and transcript information.

        Example:

            ... create gene_info_df, which has "gene_id"s
            ... create transcript_info_df, which has "transcript_id"s

            transcript_ids = transcript_info_df['transcript_id']
            mapping_df = mygene_utils.get_gene_identifier_mapping(transcript_ids)

            middle_join = gene_info_df.merge(mapping_df, on='gene_id')
            gene_and_transcript_info_df = middle_join.merge(transcript_info_df, on='transcript_id')

        Args:
            transcript_ids (list-like): a list of Ensembl transcript identifers,
                such as "ENSMUST00000049208"

        Returns:
            pd.DataFrame: a data frame mapping from the transcript to gene ids.
                columns:
                    gene_id
                    transcript_id

        Imports:
            misc.parallel
            misc.utils
            mygene
            pandas
    """
    import mygene
    import misc.parallel as parallel
    import misc.utils as utils
    import pandas as pd

    msg = ("[mygene_utils.get_transcript_to_gene_mapping]: Use the pyensembl "
        "package to find this mapping. It works locally and is much faster.")
    raise DeprecationWarning(msg)

    MG =  mygene.MyGeneInfo()
    res = MG.querymany(transcript_ids, returnall=False, 
        scopes='ensembl.transcript', fields='ensembl.gene', species='all')

    id_mapping = parallel.apply_iter_simple(res, 
        _parse_transcript_to_gene_query_result, progress_bar=True)
    id_mapping_flat = utils.flatten_lists(id_mapping)
    id_mapping_df = pd.DataFrame(id_mapping_flat)
    id_mapping_df = id_mapping_df.drop_duplicates()
    id_mapping_df = id_mapping_df.reset_index(drop=True)

    return id_mapping_df

def _parse_gene_to_transcript_query_result(entry):
    """ This helper function parses the results from the 
        get_gene_to_transcript_mapping query.

        It is not intended for external use.
    """ 
    import misc.utils as utils

    transcript = None
    gene = entry['query']
    ret = []
    
    if 'ensembl.transcript' in entry:
        transcripts = entry['ensembl.transcript']        
        transcripts = utils.wrap_string_in_list(transcripts)
        
        ret = [
            {'transcript_id': transcript, 'gene_id': gene}
                for transcript in transcripts
        ]

    elif 'ensembl' in entry:
        ensembl_entry = entry['ensembl']
        if 'transcript' in ensembl_entry:
            transcripts = ensembl_entry['transcript']
            transcripts = utils.wrap_string_in_list(transcripts)
                    
            ret = [
                {'transcript_id': transcript, 'gene_id': gene}
                    for transcript in transcripts
            ]
        
    return ret

def get_gene_to_transcript_mapping(gene_ids, mygene_url="http://mygene.info/v3"):
    """ This function uses mygene.info to find the Ensembl transcript identifiers
        which match the provided list of Ensembl gene identifiers. The result is
        returned as a pandas data frame.

        Args:
            gene_ids (list-like): a list of Ensembl gene identifiers, such as
                "ENSMUSG00000025902"

        Returns:
            pd.DataFrame a data frame mapping between transcript and gene ids.
                columns:
                    gene_id
                    transcript_id

        Imports:
            misc.parallel
            misc.utils
            mygene
            pandas
    """
    import mygene
    import misc.parallel as parallel
    import misc.utils as utils
    import pandas as pd

    msg = ("[mygene_utils.get_gene_to_transcript_mapping]: Use the pyensembl "
        "package to find this mapping. It works locally and is much faster.")
    raise DeprecationWarning(msg)

    MG =  mygene.MyGeneInfo()
    MG.url = mygene_url
    res = MG.querymany(gene_ids, returnall=False, 
        fields='ensembl.transcript', scopes='ensembl.gene', species='all')

    id_mapping = parallel.apply_iter_simple(res, 
        _parse_gene_to_transcript_query_result, progress_bar=True)
    id_mapping_flat = utils.flatten_lists(id_mapping)
    id_mapping_df = pd.DataFrame(id_mapping_flat)
    id_mapping_df = id_mapping_df.drop_duplicates()
    id_mapping_df = id_mapping_df.reset_index(drop=True)

    return id_mapping_df

