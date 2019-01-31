#! /bin/bash

CLUSTER_DEF='{"ps": ["ps.local:7000"], "workers": ["a.workers.local:7000", "b.workers.local:7000"]}' # TODO: You need to adapt this line to your own cluster, and run this script on 'ps.local'.
CLUSTER_PID=0
RUNNING_PID=0

function start_cluster {
	python3 deploy.py --cluster "${CLUSTER_DEF}" --deploy --id "ps:0" --omit&
	CLUSTER_PID=$!
	trap run_abort TERM INT
}

function stop_cluster {
	kill -s 2 ${CLUSTER_PID}
	wait ${CLUSTER_PID}
	wait ${CLUSTER_PID}
}

function run {
	local NAME=E=${1}-R=${2}-N=${3}-F=${4}-B=${5}
	python3 runner.py \
		--server "${CLUSTER_DEF}" \
		--experiment ${1} \
		--aggregator ${2} \
		--nb-workers ${3} \
		--nb-decl-byz-workers ${4} \
		--experiment-args "batch-size:${5}" \
		--max-step ${6} \
		--stdout-to ${NAME}.stdout \
		--stderr-to ${NAME}.stderr \
		--evaluation-period -1 \
		--checkpoint-period 600 \
		--summary-period -1 \
		--evaluation-delta 1000 \
		--checkpoint-delta -1 \
		--summary-delta 1000 \
		--ev-job-name ps \
		--no-wait&
	RUNNING_PID=$!
	wait ${RUNNING_PID}
}

function run_abort {
	kill -s 2 ${RUNNING_PID}
	wait ${RUNNING_PID}
	wait ${RUNNING_PID}
	stop_cluster
	exit 0
}

start_cluster
# Begin experiments
run mnist average 2 0 50 100000
# End experiments
stop_cluster
