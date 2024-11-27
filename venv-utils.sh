# >>> venv-utils
# Tools for Python virtual environments
export VENVROOT=/data/$USER/venvs
venv-list(){
    echo "Python virtual environments:"
    if [ "$1" == "-s" ]; then
        du -h --max-depth=1 $VENVROOT
    else
        ls -1 $VENVROOT
    fi
}
venv-create(){
    python3 -m venv $VENVROOT/$1
    echo "Environment created: $VENVROOT/$1 $2"
    if [ "$2" = "--upgrade-pip" ]; then
        venv-activate $1
        python -m pip install --upgrade pip
        deactivate
    fi
}
venv-remove(){
    echo "Removing environment: $VENVROOT/$1"
    deactivate
    rm -r $VENVROOT/$1
}
venv-activate(){
    source $VENVROOT/$1/bin/activate
}
alias venv-deactivate="deactivate"
# <<< venv-utils
