Bootstrap: docker
From: frankierr/skelshop:latest

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    source /.skelshop_env
