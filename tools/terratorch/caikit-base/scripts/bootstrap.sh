# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


EXTENSION_DIR=$(ls -1 */config/config.yml | grep -Ev "build/|site-packages/" | cut -d '/' -f 1 | cut -d ':' -f 2)
DEST_DIR=models
echo MODEL=$MODEL
echo MM_MODEL_ID=$MM_MODEL_ID
echo DEST_DIR=$DEST_DIR
echo EXTENSION_DIR=$EXTENSION_DIR

python3 <<EOF
#!/usr/bin/env python3
import importlib
EXTENSION = importlib.import_module("$EXTENSION_DIR")
EXTENSION.blocks.BOOTSTRAP("$MODEL").save("$DEST_DIR/$MM_MODEL_ID")
EOF

set -x
ls $DEST_DIR/$MM_MODEL_ID
