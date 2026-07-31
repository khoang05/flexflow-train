#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_NCCL_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_NCCL_TASK_H

#include "kernels/device.h"
#include <nccl.h>
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include <string>
#include <cstddef>

namespace FlexFlow {

ncclResult_t run_nccl_all_reduce(void const *send_buffer,
                                void *receive_buffer,
                                size_t count,
                                ncclDataType_t data_type,
                                ncclRedOp_t reduction_op,
                                ncclComm_t communicator,
                                ffStream_t stream);

ncclResult_t run_nccl_broadcast(void const *send_buffer,
                                void *receive_buffer,
                                size_t count,
                                ncclDataType_t data_type,
                                ncclRedOp_t reduction_op,
                                ncclComm_t communicator,
                                ffStream_t stream);

ncclResult_t run_nccl_reduce(void const *send_buffer,
                                void *receive_buffer,
                                size_t count,
                                ncclDataType_t data_type,
                                ncclRedOp_t reduction_op,
                                ncclComm_t communicator,
                                ffStream_t stream);

void nccl_task_body(void const *args,
    size_t arglen,
    void const *userdata,
    size_t userdata_len,
    Realm::Processor proc);

Realm::Event spawn_nccl_task(RealmContext &ctx,
    Realm::Processor target_proc,
    std::string const &message,
    Realm::Event precondition);

}

#endif
