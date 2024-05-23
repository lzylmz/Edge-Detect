compile: program run

program: sequential mpi

sequential: seq_main.c

	mpicc -o seq_main seq_main.c -lm

mpi: mpi_main.c

	mpicc -o mpi_main mpi_main.c -lm ./stb_image.h ./stb_image_write.h

run:

	./seq_main papagan.JPG seq_output.jpg
	mpirun -np 1 mpi_main papagan.JPG mpi1_output.jpg
	mpirun -np 2 mpi_main papagan.JPG mpi2_output.jpg
	mpirun -np 4 mpi_main papagan.JPG mpi4_output.jpg

