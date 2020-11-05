Bootstrap: docker
From: frankierr/skelshop:focal

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    source /.skelshop_env
