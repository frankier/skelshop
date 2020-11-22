Bootstrap: docker
From: frankierr/skelshop:focal_nvcaffe

%post
    echo "Nothing to do"

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    source /.skelshop_env
