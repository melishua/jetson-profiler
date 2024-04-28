#!/bin/bash
# This script runs Docker commands with sudo privileges.

case "$1" in
    get_container_id)
        sudo docker ps --format '{{.ID}}' | head -n 1
        ;;
    check_file)
        # $2 is container_id, $3 is file_path
        sudo docker exec $2 test -f $3
        ;;
    *)
        echo "Invalid command"
        exit 1
        ;;
esac
