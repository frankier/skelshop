Bootstrap: docker
From: frankierr/skelshop:focal_nvcaffe

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    source /.skelshop_env
