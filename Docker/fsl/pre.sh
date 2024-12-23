#! bin/bash

# Por el amor de dios recordar que esto debe ser LF y no CRLF!!!!!!!!!!!!

if [ -z "$(ls -A /output)" ]; then 
    echo "Output directory is empty. Wait until dcm2bids finishes and run the compose again :_("
else 

        anat_new_value=$(jq -r '.anat.value' /config/parameters.json)
        echo "Starting preprocessing >:/";
        echo "First, the brain only UwU";

        for patient_dir in /output/derivatives/sub-*; do
            patient_num=$(basename $patient_dir)
            echo "Preprocessing ${patient_dir}"

            for session_dir in ${patient_dir}/ses-*; do
                session_num=$(basename $session_dir)
                echo "Processing session ${session_num}"

                if [ -d ${session_dir}/anat ]; then

                    mkdir -p /output/derivatives/preprocessed/${patient_num}/${session_num}/anat
                    for file in ${session_dir}/anat/*.nii.gz*; do
                        base=$(basename $file)
                        echo "Registering to MNI :)"
                        flirt -in $file \
                              -ref /usr/local/fsl/data/standard/MNI152_T1_2mm_brain \
                              -out /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_brain_registered \
                              -omat /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_brain_registered.mat \
                              -bins 256 \
                              -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 \
                              -interp trilinear
                        echo "Finished UwU"
                    done
                fi
            done    
        done

       for patient_dir in /output/sub-*; do
        patient_num=$(basename $patient_dir)
        echo "Preprocessing ${patient_dir}"

        for session_dir in ${patient_dir}/ses-*; do
            session_num=$(basename $session_dir)
            echo "Processing session ${session_num}"

            if [ -d ${session_dir}/anat ]; then

                mkdir -p /output/derivatives/preprocessed/${patient_num}/${session_num}/anat
                for file in ${session_dir}/anat/*.nii.gz*; do
                    base=$(basename $file)
                    #echo "Cutting neck :/"
                    #robustfov -i  $file -r /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_robust.nii.gz
                    echo "Skull stripping :C"
                    bet $file /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_brain -f ${anat_new_value} -g 0

                    echo "Registering to MNI :)"
                    flirt -in /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_brain\
                          -ref /usr/local/fsl/data/standard/MNI152_T1_2mm_brain \
                          -out /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_brain_registered \
                          -omat /output/derivatives/preprocessed/${patient_num}/${session_num}/anat/${base%.nii.gz}_brain_registered.mat \
                          -bins 256 \
                          -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 \
                          -interp trilinear
                    
                    echo "Finished UwU"
                done
            fi 
            if [ -d ${patient_dir}/func ]; then \
                mkdir -p /output/derivatives/preprocessed/${patient_num}/func && \
                func_new_value=$(jq -r '.func.value' /config/parameters.json); \
                for file in ${patient_dir}/func/*.nii.gz*; do \
                    base=$(basename $file); \
                    bet $file /output/derivatives/preprocessed/${patient_num}/func/${base%.nii.gz}_brain -F -f ${func_new_value} -g 0; \
                    rm -f /output/derivatives/preprocessed/${patient_num}/func/${base%.nii.gz}_brain_mask.nii.gz; \
                done
            fi
            done
            echo "Preprocessing for patient ${patient_num} done :)" 
        done
        echo "Preprocessing finished, have a nice day ;)" 
                
fi