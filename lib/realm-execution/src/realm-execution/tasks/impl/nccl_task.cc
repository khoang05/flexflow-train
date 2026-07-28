#include "realm-execution/tasks/impl/nccl_task.h"
#include "kernels/device.h"
#include "realm-execution/tasks/impl/nccl_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_nccl_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.h"

#include <cstdio>
#include <nccl.h>

namespace FlexFlow {

ncclResult_t run_nccl_all_reduce(void const *send_buffer,
                                void *receive_buffer,
                                size_t count,
                                ncclDataType_t data_type,
                                ncclRedOp_t reduction_op,
                                ncclComm_t communicator,
                                ffStream_t stream) {
  return ncclAllReduce(send_buffer,
                       receive_buffer,
                       count,
                       data_type,
                       reduction_op,
                       communicator,
                       stream);
}

void nccl_task_body(void const *args,
                    size_t arglen,
                    void const *userdata,
                    size_t userdata_len,
                    Realm::Processor proc) {
  (void)userdata;
  (void)userdata_len;
  (void)proc;

  NcclTaskArgs task_args = nccl_task_args_from_serializable(
      deserialize_task_args<SerializableNcclTaskArgs>(args, arglen));

  int nccl_version = 0;
  ncclResult_t result = ncclGetVersion(&nccl_version);

  if (result != ncclSuccess) {
    std::printf("NCCL error: %s\n", ncclGetErrorString(result));
    return;
  }

  std::printf("%s\n", task_args.message.c_str());
  std::printf("NCCL version: %d\n", nccl_version);
}

Realm::Event spawn_nccl_task(RealmContext &ctx,
                             Realm::Processor target_proc,
                             std::string const &message,
                             Realm::Event precondition) {
  NcclTaskArgs task_args = NcclTaskArgs{
      /*message=*/message,
  };

  std::string serialized_args =
      serialize_task_args(nccl_task_args_to_serializable(task_args));

  return ctx.spawn_task(
      target_proc,
      task_id_t::NCCL_HELLO_WORLD_TASK_ID,
      serialized_args.data(),
      serialized_args.size(),
      Realm::ProfilingRequestSet{},
      precondition);
}

} // namespace FlexFlow
