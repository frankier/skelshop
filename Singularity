Bootstrap: docker
From: frankierr/skelshop:focal_nvcaffe

%script
    echo "Nothing to do"

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    source /.skelshop_env
