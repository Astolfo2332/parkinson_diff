if [ -z "$(ls -A /output)" ]; then \
        echo "Output directory is empty. Wait until dcm2bids finishes and run the compose again :_(" ;\
    else \
        echo "Starting preprocessing >:/" ;\
        rm -rf output/tmp_dcm2bids && \
        for patient_dir in /output/sub-*; do \
            patient_num=$(basename $patient_dir); \
            if [ -d ${patient_dir}/anat ]; then \
                mkdir -p /output/derivatives/preprocessed/${patient_num}/anat && \
                anat_new_value=$(jq -r '.anat.value' /config/parameters.json); \
                for file in ${patient_dir}/anat/*.nii.gz*; do \
                    base=$(basename $file); \
                    bet $file /output/derivatives/preprocessed/${patient_num}/anat/${base%.nii.gz}_brain -f ${anat_new_value} -g 0; \
                done; \
            fi; \
            if [ -d ${patient_dir}/func ]; then \
                mkdir -p /output/derivatives/preprocessed/${patient_num}/func && \
                func_new_value=$(jq -r '.func.value' /config/parameters.json); \
                for file in ${patient_dir}/func/*.nii.gz*; do \
                    base=$(basename $file); \
                    bet $file /output/derivatives/preprocessed/${patient_num}/func/${base%.nii.gz}_brain -F -f ${func_new_value} -g 0; \
                    rm -f /output/derivatives/preprocessed/${patient_num}/func/${base%.nii.gz}_brain_mask.nii.gz; \
                done; \
            fi; \
            echo "Preprocessing for patient ${patient_num} done :)" ;\
        done && \
        echo "Preprocessing finished, have a nice day ;)" ;\
                
    fi;