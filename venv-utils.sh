# >>> venv-utils
# Tools for Python virtual environments
export VENVROOT=$HPCPERM/venvs
export PIPCACHEXL=$SCRATCH/cache/pip
venv-list(){
    # Virtual environment listing tool.
    #
    # Usage:
    #   venv-list [-s]
    #
    # Examples:
    #   venv-list       # List all venvs
    #   venv-list -s    # List all venvs with their size and full paths
    #
    echo "Python virtual environments:"
    if [ "$1" == "-s" ]; then
        du -h --max-depth=1 $VENVROOT
    else
        ls -1 $VENVROOT
    fi
}
venv-create(){
    # Virtual environment creation tool.
    #
    # Usage:
    #   venv-create NAME [PYTHON [--upgrade-pip]]
    #
    #   * NAME: name of the environment to be created
    #   * PYTHON: the Python interpreter to use (default is python3)
    #   * --upgrade-pip: flag to trigger a pip upgrade
    #
    # Examples:
    #   venv-create for-test1
    #   venv-create for-test2 python3.11 --upgrade-pip
    #
    PYTHON=${2:-python3}
    $PYTHON -m venv $VENVROOT/$1
    if [ -d $VENVROOT/$1 ]; then
        echo "Environment created: $VENVROOT/$1 $2 $3"
        if [ "$3" == "--upgrade-pip" ]; then
            venv-activate $1
            python -m pip install --upgrade pip
            deactivate
        fi
    else
        echo "No environment created. Args: $1 $2 $3"
    fi
}
venv-remove(){
    # Virtual environment removal tool.
    #
    # Usage:
    #   venv-remove NAME 
    #
    echo "Removing environment: $VENVROOT/$1"
    deactivate
    rm -rf $VENVROOT/$1
}
venv-activate(){
    # Virtual environment activation tool.
    #
    # Usage:
    #   venv-activate NAME
    #
    source $VENVROOT/$1/bin/activate
}
alias venv-deactivate="deactivate"
# <<< venv-utils
