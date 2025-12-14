#!/usr/bin/env bash
set -euo pipefail

#
#  From fps/:
#    $ nohup stdbuf -oL bash fps_sweep_scipy.sh > fps_sweep_scipy.txt 2>&1 &
#
#    $ PYTHONPATH=.. uv run python -m scipyturbo.scipy_simulator 512
#

rm -f fps_N*.log
OUT_CSV="fps_sweep_scipy.csv"
rm -f "${OUT_CSV}"

# Header
echo "N,FPS" > "${OUT_CSV}"

for K in $(seq 5 12); do
    N=$((2 ** K))

    LOG="fps_N${N}.log"
    echo "Running N=${N} ..."

    # Run and capture full output (run from fps/, so PYTHONPATH=..)
    stdbuf -oL env PYTHONPATH=.. uv run python -m scipyturbo.scipy_simulator "${N}" | tee "${LOG}"

    # Extract FPS from: "Frames per second (FPS)                 =      73.8932"
    FPS=$(grep "Frames per second (FPS)" "${LOG}" | tail -n 1 | awk '{print $NF}')

    echo "${N},${FPS}" >> "${OUT_CSV}"
done

echo "Done. Results written to ${OUT_CSV}"