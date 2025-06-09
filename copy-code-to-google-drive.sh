#!/usr/bin/env bash


rclone copy --interactive --filter-from=./rclone-filter-file.txt "." "gdrive-notebooks:MNLP - RAG/"
