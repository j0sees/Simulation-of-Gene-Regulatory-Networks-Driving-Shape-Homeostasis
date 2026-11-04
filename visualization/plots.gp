set ylabel "time (s)"
set xlabel "# processes"
set title "Average time per generation"

#set terminal png enhanced font "Helvetica,12"
#set output 'avgtime_per_gen_nProcs_vs_ChunkSize.png'

#plot "test_results_nProcs_vs_ChunkSize.stats" index 0 u 1:4 t "chunkSize 1" w lp, "" index 1 u 1:4 t "chunkSize 2" w lp, "" index 2 u 1:4 t "chunkSize 3" w lp, "" index 3 u 1:4 t "chunkSize 4" w lp, "" index 4 u 1:4 t "chunkSize 5" w lp, "" index 5 u 1:4 t "chunkSize default" w lp
#plot "test_final_mod.stats" index 0 u 2:4 t "7 procs" w lp, "" index 2 u 2:4 t "10 procs" w lp, "" index 3 u 2:4 t "11 procs" w lp, "" index 4 u 2:4 t "12 procs" w lp, "" index 5 u 2:4 t "13 procs" w lp, "" index 6 u 2:4 t "14 procs" w lp, "" index 7 u 2:4 t "15 procs" w lp, "" index 8 u 2:4 t "16 procs" w lp, "" index 9 u 2:4 t "17 procs" w lp, "" index 10 u 2:4 t "18 procs" w lp, "" index 11 u 2:4 t "19 procs" w lp, "" index 1 u 2:4 t "20 procs" w lp
#plot "test_results_pop_50.stats" u 1:4 t "50 individuals" w lp
pause -1
