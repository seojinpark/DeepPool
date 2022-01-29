#include "streamingDataset.h"

#include <torch/torch.h>
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <filesystem>
#include <fcntl.h>

#include "hdf5.h"

#define MAX_FILE_NAME_LEN 128
#define NUM_FILE_STRINGS 8
#define MAX_STRING_SIZE 1024
#define PORT 9008
std::vector<int> pids;

void signalHandler(int signum)
{
    std::cout << "Interrupt signal (" << signum << ") received.\n";

    for (auto pid : pids)
        kill(pid, SIGKILL);

    exit(signum);
}

uint64_t ReadXBytes(int socket, void *buffer, uint64_t x)
{
    uint64_t bytesRead = 0;
    int result;
    while (bytesRead < x)
    {
        result = read(socket, buffer + bytesRead, x - bytesRead);
        if (result < 1)
        {
            printf("\nError receiving from server %d\n", result);
            return 0;
        }
        bytesRead += result;
    }
    return bytesRead;
}

int64_t HDF5_get_dataset_len(hid_t h5FileID, const char *datasetName)
{
    hid_t space, dset; /* Handles */

    dset = H5Dopen(h5FileID, datasetName, H5P_DEFAULT);

    space = H5Dget_space(dset);
    const int ndims = H5Sget_simple_extent_ndims(space);

    hsize_t dims[ndims] = {0};
    memset(dims, 0, sizeof(hsize_t) * ndims);
    int status_n = H5Sget_simple_extent_dims(space, dims, NULL);

    assert(ndims == status_n);

    uint64_t nElements = dims[0];
    for (int n = 1; n < ndims; n++)
    {
        nElements *= dims[n];
    }

    H5Dclose(dset);
    H5Sclose(space);
    return nElements;
}

size_t get_dataset_datatype_size(hid_t h5FileID, const char *datasetName)
{
    hid_t filetype, dset;
    herr_t status;
    size_t sdim;

    dset = H5Dopen(h5FileID, datasetName, H5P_DEFAULT);
    filetype = H5Dget_type(dset);
    sdim = H5Tget_size(filetype);
    sdim++; /* Make room for null terminator */

    status = H5Dclose(dset);
    status = H5Tclose(filetype);
    if (status < 0)
        throw std::runtime_error("hdf5 error\n");
    return sdim;
}

void parse_HDF5_str_dataset(hid_t h5FileID, const char *datasetName, void *buf, size_t *strLen = NULL)
{
    std::vector<std::string> iid_strings;
    herr_t status;
    hid_t memtype, space, dset;
    size_t sdim;

    dset = H5Dopen(h5FileID, datasetName, H5P_DEFAULT);

    space = H5Dget_space(dset);
    const int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[ndims] = {0};
    memset(dims, 0, sizeof(hsize_t) * ndims);
    H5Sget_simple_extent_dims(space, dims, NULL);

    sdim = get_dataset_datatype_size(h5FileID, datasetName);

    if (strLen != NULL)
        *strLen = sdim;

    /*
     * Create the memory datatype.
     */
    memtype = H5Tcopy(H5T_C_S1);
    if (H5Tset_size(memtype, sdim) < 0)
    {
        throw std::runtime_error("Error H5Tset_size\n");
    }

    /*
     * Read the data.
     */
    if (H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
    {
        throw std::runtime_error("Error reading hdf5 file\n");
    }

    status = H5Dclose(dset);
    status = H5Sclose(space);
    status = H5Tclose(memtype);
    if (status < 0)
        throw std::runtime_error("hdf5 error\n");
}

// std::vector<std::string> parse_HDF5_str_dataset(hid_t h5FileID, const char * datasetName){
//     std::vector<std::string> iid_strings;

//     hid_t       filetype, memtype, space, dset;
//     hsize_t     dims[1] = {64};
//     size_t      sdim;
//     // char        wdata[DIM0][SDIM] = {"Parting", "is such", "sweet", "sorrow."},
//                                             /* Write buffer */
//     char        **rdata;                    /* Read buffer */

//     dset = H5Dopen (h5FileID, datasetName, H5P_DEFAULT);
//     filetype = H5Dget_type (dset);
//     sdim = H5Tget_size (filetype);
//     sdim++;                         /* Make room for null terminator */

//     /*
//      * Get dataspace and allocate memory for read buffer.  This is a
//      * two dimensional dataset so the dynamic allocation must be done
//      * in steps.
//      */
//     space = H5Dget_space (dset);
//     H5Sget_simple_extent_dims (space, dims, NULL);

//     /*
//      * Allocate array of pointers to rows.
//      */
//     rdata = (char **) malloc (dims[0] * sizeof (char *));

//     /*
//      * Allocate space for integer data.
//      */
//     rdata[0] = (char *) malloc (dims[0] * sdim * sizeof (char));

//     /*
//      * Set the rest of the pointers to rows to the correct addresses.
//      */
//     for (hsize_t i=1; i<dims[0]; i++)
//         rdata[i] = rdata[0] + i * sdim;

//     /*
//      * Create the memory datatype.
//      */
//     memtype = H5Tcopy (H5T_C_S1);
//     H5Tset_size (memtype, sdim);

//     /*
//      * Read the data.
//      */
//     if(H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata[0]) < 0){
//         throw std::runtime_error("Error reading hdf5 file\n");
//     }

//     /*
//      * Output the data to the screen.
//      */
//     for (hsize_t i=0; i<dims[0]; i++){
//         iid_strings.emplace_back(std::string(rdata[i]));
//         // printf ("%s[%d]: %s\n", datasetName, i, iid_strings[i].c_str());
//     }

//     free (rdata[0]);
//     free (rdata);
//     H5Dclose (dset);
//     H5Sclose (space);

//     return iid_strings;
// }

// torch::Tensor
uint64_t parse_HDF5_llong_dataset(hid_t h5FileID, const char *datasetName, void *buf, uint64_t *dims)
{
    hid_t space, dset; /* Handles */
    herr_t status;

    /*
     * Open file and dataset.
     */
    dset = H5Dopen(h5FileID, datasetName, H5P_DEFAULT);

    /*
     * Get dataspace and allocate memory for read buffer.  This is a
     * two dimensional dataset so the dynamic allocation must be done
     * in steps.
     */
    space = H5Dget_space(dset);
    const int ndims = H5Sget_simple_extent_ndims(space);

    // hsize_t dims[ndims] = {0};
    memset(dims, 0, sizeof(hsize_t) * ndims);
    int status_n = H5Sget_simple_extent_dims(space, (hsize_t *)dims, NULL);

    assert(ndims == status_n);

    /*
     * Allocate buffer
     */
    uint64_t nElements = dims[0];
    for (int n = 1; n < ndims; n++)
    {
        nElements *= dims[n];
    }

    /*
     * Read the data.
     */
    if (H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
    {
        throw std::runtime_error("Error reading hdf5 file\n");
    }

    status = H5Dclose(dset);
    status = H5Sclose(space);
    if (status < 0)
        throw std::runtime_error("hdf5 error\n");
    return ndims;
}

void parse_HDF5_llong_scalar_dataset(hid_t h5FileID, const char *datasetName, int64_t *buf)
{
    hid_t space, dset; /* Handles */
    herr_t status;

    /*
     * Open file and dataset.
     */
    dset = H5Dopen(h5FileID, datasetName, H5P_DEFAULT);

    /*
     * Get dataspace and allocate memory for read buffer.  This is a
     * two dimensional dataset so the dynamic allocation must be done
     * in steps.
     */
    space = H5Dget_space(dset);

    /*
     * Read the data.
     */
    if (H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
    {
        throw std::runtime_error("Error reading hdf5 file\n");
    }

    /*
     * Close and release resources.
     */
    status = H5Dclose(dset);
    status = H5Sclose(space);
    if (status < 0)
        throw std::runtime_error("hdf5 error\n");
}

uint64_t parse_HDF5_float_dataset(hid_t h5FileID, const char *datasetName, void *buf, uint64_t *dims)
{
    hid_t space, dset; /* Handles */
    herr_t status;

    /*
     * Open file and dataset.
     */
    dset = H5Dopen(h5FileID, datasetName, H5P_DEFAULT);

    /*
     * Get dataspace and allocate memory for read buffer.  This is a
     * two dimensional dataset so the dynamic allocation must be done
     * in steps.
     */
    space = H5Dget_space(dset);
    const int ndims = H5Sget_simple_extent_ndims(space);

    // hsize_t dims[ndims] = {0};
    memset(dims, 0, sizeof(hsize_t) * ndims);
    int status_n = H5Sget_simple_extent_dims(space, (hsize_t *)dims, NULL);

    assert(ndims == status_n);

    /*
     * Allocate buffer
     */
    uint64_t nElements = dims[0];
    for (int n = 1; n < ndims; n++)
    {
        nElements *= dims[n];
    }

    /*
     * Read the data.
     */
    if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
    {
        throw std::runtime_error("Error reading hdf5 file\n");
    }

    status = H5Dclose(dset);
    status = H5Sclose(space);
    if (status < 0)
        throw std::runtime_error("hdf5 error\n");

    // return theArray;
    return ndims;
}

void *create_shared_memory(size_t size)
{
    // Our memory buffer will be readable and writable:
    int protection = PROT_READ | PROT_WRITE;

    // The buffer will be shared (meaning other processes can access it), but
    // anonymous (meaning third-party processes cannot obtain an address for it),
    // so only this process and its children will be able to use it:
    int visibility = MAP_SHARED | MAP_ANONYMOUS;

    // The remaining parameters to `mmap()` are not important for this use case,
    // but the manpage for `mmap` explains their purpose.
    return mmap(NULL, size, protection, visibility, -1, 0);
}

int get_batch_filename(void *buffer, int rank = -1, int worldSize = -1, std::deque<std::string> *hdf5_file_vec = NULL)
{
    std::string path = "/dev/shm/batches";
    int64_t minBatchIndex = -1;
    std::string minBatchFile;
    std::filesystem::path lockPathExt = std::filesystem::path(".lock");
    while (minBatchIndex == -1)
    {
        for (const auto &entry : std::filesystem::directory_iterator(path))
        {
            std::filesystem::path lockPath = std::filesystem::path(entry).replace_extension(lockPathExt);
            if (entry.path().extension().compare(".hdf5") == 0 && !std::filesystem::exists(lockPath))
            {

                std::string fileName = entry.path().stem();
                int batchIndex = atoi(fileName.substr(6).c_str());

                if (rank > -1 && worldSize > -1)
                {
                    assert(rank < worldSize);
                    if (batchIndex - rank >= 0 && batchIndex % worldSize == rank)
                    {
                        if (hdf5_file_vec != NULL)
                            hdf5_file_vec->push_back(entry.path());
                        if (minBatchIndex == -1 || batchIndex < minBatchIndex)
                        {
                            minBatchIndex = batchIndex;
                            minBatchFile = entry.path();
                        }
                    }
                }
                else if (minBatchIndex == -1 || batchIndex < minBatchIndex)
                {
                    minBatchIndex = batchIndex;
                    minBatchFile = entry.path();
                }
            }
        }
    }
    // std::cout << "found! " << fileName << " " << fileName.substr(6).c_str()<< " " << batchIndex << '\n';
    // std::cout << "minBatchIndex " << minBatchIndex << " minBatchFile " << minBatchFile << std::endl;
    if (hdf5_file_vec == NULL)
        memcpy(buffer, minBatchFile.c_str(), minBatchFile.size());
    return 0;
}

void init_sharedBuffers(sharedBuffers *sbufs, char *hdf5_file)
{
    auto batchFile = H5Fopen(hdf5_file, H5F_ACC_RDONLY, H5P_DEFAULT);

    parse_HDF5_llong_scalar_dataset(batchFile, "/len", &sbufs->batchSize);
    parse_HDF5_llong_scalar_dataset(batchFile, "/index", &sbufs->index);

    // sbufs->iidVecLen = HDF5_get_dataset_len(batchFile, "/iid");
    sbufs->weightsLen = HDF5_get_dataset_len(batchFile, "/weights");
    sbufs->imagesLen = HDF5_get_dataset_len(batchFile, "/images");
    sbufs->labelsLen = HDF5_get_dataset_len(batchFile, "/labels");

    // sbufs->iidVecStrLen = get_dataset_datatype_size(batchFile, "/iid");
    // sbufs->iidVec = (char *)create_shared_memory(sizeof(char)*sbufs->iidVecLen*MAX_STRING_SIZE+1);

    sbufs->weights = (void *)create_shared_memory(sizeof(float) * sbufs->weightsLen);
    sbufs->images = (void *)create_shared_memory(sizeof(float) * sbufs->imagesLen);
    sbufs->labels = (void *)create_shared_memory(sizeof(int64_t) * sbufs->labelsLen);

    // parse_HDF5_str_dataset(batchFile, "/iid", (char *)sbufs->iidVec);
    sbufs->weightsNDims = parse_HDF5_float_dataset(batchFile, "/weights", sbufs->weights, sbufs->weightsDims);
    sbufs->imagesNDims = parse_HDF5_float_dataset(batchFile, "/images", sbufs->images, sbufs->imagesDims);
    sbufs->labelsNDims = parse_HDF5_llong_dataset(batchFile, "/labels", sbufs->labels, sbufs->labelsDims);

    H5close();
}

void init_shared_mutex(sharedBuffers *sbufs)
{
    int mutex_fd;
    int mode = S_IRWXU | S_IRWXG;

    mutex_fd = shm_open(sbufs->mutexID, O_CREAT | O_RDWR, mode);

    if (mutex_fd < 0)
    {
        perror("failure on shm_open on mutex_fd");
        exit(1);
    }

    if (ftruncate(mutex_fd, sizeof(pthread_mutex_t)) == -1)
    {
        perror("Error on ftruncate to sizeof pthread_mutex_t\n");
        exit(-1);
    }

    sbufs->mp_mutex = (pthread_mutex_t *)mmap(NULL, sizeof(pthread_mutex_t),
                                              PROT_READ | PROT_WRITE, MAP_SHARED, mutex_fd, 0);

    if (sbufs->mp_mutex == MAP_FAILED)
    {
        perror("Error on mmap on mutex\n");
        exit(1);
    }
    close(mutex_fd);

    /* set mutex shared between processes */
    pthread_mutexattr_setpshared(&sbufs->mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(sbufs->mp_mutex, &sbufs->mutexAttr);
}

void free_sharedBuffers(sharedBuffers *sbufs)
{
    // munmap(sbufs->iidVec, sizeof(char)*sbufs->iidVecLen*MAX_STRING_SIZE+1);
    munmap(sbufs->weights, sizeof(float) * sbufs->weightsLen);
    munmap(sbufs->images, sizeof(float) * sbufs->imagesLen);
    munmap(sbufs->labels, sizeof(int64_t) * sbufs->labelsLen);
    munmap(sbufs, sizeof(sharedBuffers));

    pthread_mutexattr_destroy(&sbufs->mutexAttr);
    pthread_mutex_destroy(sbufs->mp_mutex);
    shm_unlink(sbufs->mutexID);
    munmap(sbufs->mp_mutex, sizeof(pthread_mutex_t));
}

StreamingDataset::StreamingDataset(int rank, int worldSize)
{
    this->rank = rank;
    this->worldSize = worldSize;

    int mutex_fd;
    int mode = S_IRWXU | S_IRWXG;
    sprintf(mutexID, "/main-mutex");

    mutex_fd = shm_open(mutexID, O_CREAT | O_RDWR, mode);

    if (mutex_fd < 0)
    {
        perror("failure on shm_open on mutex_fd");
        exit(1);
    }

    if (ftruncate(mutex_fd, sizeof(pthread_mutex_t)) == -1)
    {
        perror("Error on ftruncate to sizeof pthread_mutex_t\n");
        exit(-1);
    }

    mp_mutex = (pthread_mutex_t *)mmap(NULL, sizeof(pthread_mutex_t),
                                       PROT_READ | PROT_WRITE, MAP_SHARED, mutex_fd, 0);

    if (mp_mutex == MAP_FAILED)
    {
        perror("Error on mmap on mutex\n");
        exit(1);
    }
    close(mutex_fd);

    /* set mutex shared between processes */
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(mp_mutex, &mutexAttr);

    uint64_t BUFFERSIZE = 1024;
    char buffer[BUFFERSIZE];

    for (uint64_t w = 0; w < numWorkers; w++)
    {
        sharedBuffers *newBufs = (sharedBuffers *)create_shared_memory(sizeof(sharedBuffers));

        memset(buffer, 0, BUFFERSIZE);
        get_batch_filename(buffer, (rank * numWorkers) + w, worldSize * numWorkers);
        init_sharedBuffers(newBufs, buffer);

        sprintf(newBufs->mutexID, "/mutex-%ld", w);
        init_shared_mutex(newBufs);

        newBufs->ready = true;
        sharedWorkerBuffs.push_back(newBufs);

        remove(buffer);
    }

    int pid = 0;
    for (uint64_t w = 0; w < numWorkers; w++)
    {
        pid = fork();
        if (pid == 0)
        {
            // printf("worker id %d starting\n", w);
            workerID = w;
            bufferIndex = w;
            workerRunMain();
            break;
        }
        else
        {
            pids.push_back(pid);
            std::thread thObj(&StreamingDataset::workerToTensorsThread, this, w);
            threads.push_back(std::move(thObj));
        }
    }

    if (pid != 0)
    {
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        signal(SIGKILL, signalHandler);
    }
};

StreamingDataset::~StreamingDataset(void)
{
    // pthread_mutexattr_destroy(&mutexAttr);
    // pthread_mutex_destroy(mp_mutex);
    // shm_unlink(MUTEX);

    for (auto sbufs : sharedWorkerBuffs)
        free_sharedBuffers(sbufs);
};

void StreamingDataset::workerRunMain(void)
{
    uint64_t BUFFERSIZE = 1024;
    char buffer[BUFFERSIZE] = {0};
    std::deque<std::string> hdf5_file_vec;
    sharedBuffers *sbufs = sharedWorkerBuffs[workerID];
    printf("worker id %ld starting\n", workerID);

    while (!sbufs->stopWorker)
    {
        // auto t1 = std::chrono::high_resolution_clock::now();

        if (hdf5_file_vec.size() == 0)
            get_batch_filename(buffer, (rank * numWorkers) + workerID, worldSize * numWorkers, &hdf5_file_vec);
        memset(buffer, 0, BUFFERSIZE);
        memcpy(buffer, hdf5_file_vec.front().c_str(), hdf5_file_vec.front().size());
        hdf5_file_vec.pop_front();

        // auto t2 = std::chrono::high_resolution_clock::now();

        while (sbufs->ready)
        {
            usleep(100); /* wait for buffer to be accessed */
        };

        // auto t3 = std::chrono::high_resolution_clock::now();

        auto batchFile = H5Fopen(buffer, H5F_ACC_RDONLY, H5P_DEFAULT);

        parse_HDF5_llong_scalar_dataset(batchFile, "/index", &sbufs->index);

        // sbufs->iidVecStrLen = get_dataset_datatype_size(batchFile, "/iid");
        // parse_HDF5_str_dataset(batchFile, "/iid", (char *)sbufs->iidVec, &sbufs->iidVecStrLen);

        sbufs->weightsNDims = parse_HDF5_float_dataset(batchFile, "/weights", sbufs->weights, sbufs->weightsDims);
        sbufs->imagesNDims = parse_HDF5_float_dataset(batchFile, "/images", sbufs->images, sbufs->imagesDims);
        sbufs->labelsNDims = parse_HDF5_llong_dataset(batchFile, "/labels", sbufs->labels, sbufs->labelsDims);

        H5close();

        sbufs->ready = true;
        remove(buffer);
        // auto t4 = std::chrono::high_resolution_clock::now();

        // if(sbufs->index % 50 == 0){
        //     // auto t2 = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        //     auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
        //     auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
        //     std::cout << workerID << " " << sbufs->index << " \t\t\t\tworkerRunMain " << duration.count() << " " \
        //         << duration2.count() << " " << duration3.count() << " microseconds/batch" << std::endl;
        // }
    }
}

void StreamingDataset::workerToTensorsThread(int wID)
{
    batchData batch;
    // static int last = 0;
    // int64_t index = -1;
    sharedBuffers *sbufs = sharedWorkerBuffs[wID];

    while (!stopDataset)
    {
        // auto t1 = std::chrono::high_resolution_clock::now();
        while (!sbufs->ready)
        {
            usleep(100);
        };
        // pthread_mutex_lock(sbufs->mp_mutex);
        // auto t2 = std::chrono::high_resolution_clock::now();
        // sbufs->ready = false;

        batch.len = sbufs->batchSize;
        batch.index = sbufs->index;

        auto options = torch::TensorOptions().dtype(torch::kFloat32); //.device(torch::kCUDA, 1);

        at::IntArrayRef weightsDimsArr((int64_t *)sbufs->weightsDims, (int64_t *)sbufs->weightsDims + sbufs->weightsNDims);
        batch.weights = torch::from_blob((void *)sbufs->weights, weightsDimsArr, options).to(torch::device({torch::kCUDA, this->rank})).set_requires_grad(true);

        at::IntArrayRef imagesDimsArr((int64_t *)sbufs->imagesDims, (int64_t *)sbufs->imagesDims + sbufs->imagesNDims);
        batch.images = torch::from_blob((void *)sbufs->images, imagesDimsArr, options).to(torch::device({torch::kCUDA, this->rank}));

        auto longOptions = torch::TensorOptions().dtype(torch::kInt64);
        at::IntArrayRef labelsDimsArr((int64_t *)sbufs->labelsDims, (int64_t *)sbufs->labelsDims + sbufs->labelsNDims);
        batch.labels = torch::from_blob((void *)sbufs->labels, labelsDimsArr, longOptions).to(torch::device({torch::kCUDA, this->rank}));

        // pthread_mutex_unlock(sbufs->mp_mutex);
        local_mutex.lock();
        readyBatches.push_back(batch);
        sbufs->ready = false;
        local_mutex.unlock();
        // if(batch.index % 50 == wID){
        //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        //     std::cout << batch.index << " workerToTensorsThread " << wID << " : " << duration.count() << " microseconds/batch" << std::endl;
        // }
        while (readyBatches.size() > 256)
        {
            usleep(100);
        }
    }
}

torch::optional<batchData> StreamingDataset::read_batch()
{
    batchData batch;
    while (1)
    {
        local_mutex.lock();
        if (readyBatches.size() > 0)
        {
            batch = readyBatches.front();
            if (batch.index % 50 == 0)
                std::cout << "readyBatches.size()  " << readyBatches.size() << std::endl;
            readyBatches.pop_front();
            local_mutex.unlock();
            break;
        }
        else
            local_mutex.unlock();
    }
    return batch;
}

int64_t StreamingDataset::init(void)
{
    return 0;
}

torch::optional<batchData> StreamingDataset::get_batch(size_t)
{
    return read_batch();
}

torch::optional<size_t> StreamingDataset::size() const
{
    return this->epochLen * 500;
}

int64_t StreamingDataset::test(void)
{
    // Generate a data loader.
    // torch::data::DataLoaderOptions Dopts(1);
    // auto data_loader = torch::data::make_data_loader<StreamingDataset>(Dopts);

    int64_t count = 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    while (1)
    {
        auto nextBatch = read_batch();
        if (count % 100 == 0)
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            std::cout << "iteration " << count << ", averate iter time: " << duration.count() / count << " microseconds/batch" << std::endl;
            // std::cout << "Starting Epoch - " << count << std::endl;
        }
        count++;
    }
}