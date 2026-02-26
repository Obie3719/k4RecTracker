SCRIPT="/work/omunkombwe/k4RecTracker/HTCondor/gun_split_submit/theta90/submit_theta90_part7.sh"
OUTDIR="/ceph/omunkombwe/files/deg90"
TAG="pi_P0.2-10.0GeV_theta90deg_seed1908_nev100_part7"

# 1) keep old files out of the way (recommended)
mkdir -p "${OUTDIR}/retry_backup_4393386"
for k in sim digi fit; do
  [ -f "${OUTDIR}/${k}_${TAG}.root" ] && mv "${OUTDIR}/${k}_${TAG}.root" "${OUTDIR}/retry_backup_4393386/"
done

# 2) submit one-off Condor job
cat > /tmp/resubmit_part7.jdl <<EOF
Universe = vanilla
initialdir = /work/omunkombwe/k4RecTracker
executable = ${SCRIPT}

request_cpus = 1
request_memory = 8000
request_disk = 1000000
+RequestWalltime = 4*3600
accounting_group = cms.higgs

log = /work/omunkombwe/k4RecTracker/HTCondor/gun_split_submit/log/retry_part7_\$(ClusterID).log
output = /work/omunkombwe/k4RecTracker/HTCondor/gun_split_submit/out/retry_part7_\$(ClusterID)_\$(ProcID).out
error = /work/omunkombwe/k4RecTracker/HTCondor/gun_split_submit/err/retry_part7_\$(ClusterID)_\$(ProcID).err
queue
EOF

condor_submit /tmp/resubmit_part7.jdl
