#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/tasks/impl/nccl_task.h"
#include "realm-execution/tasks/impl/nccl_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_nccl_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "task-spec/permissions.h"
#include "utils/containers/map_values.h"
#include "utils/optional.h"

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

ncclResult_t run_nccl_broadcast(void const *send_buffer,
                                void *receive_buffer,
                                size_t count,
                                ncclDataType_t data_type,
                                int root_rank,
                                ncclComm_t communicator,
                                ffStream_t stream) {
  return ncclBroadcast(send_buffer,
                       receive_buffer,
                       count,
                       data_type,
                       root_rank,
                       communicator,
                       stream);
}

ncclResult_t run_nccl_reduce(void const *send_buffer,
                             void *receive_buffer,
                             size_t count,
                             ncclDataType_t data_type,
                             ncclRedOp_t reduction_op,
                             int root_rank,
                             ncclComm_t communicator,
                             ffStream_t stream) {
  return ncclReduce(send_buffer,
                    receive_buffer,
                    count,
                    data_type,
                    reduction_op,
                    root_rank,
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

NCCLTaskArgs task_args = nccl_task_args_from_serializable(
    deserialize_task_args<SerializableNcclTaskArgs>(args, arglen));

  RealmContext ctx{proc};

  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_ff_handle(
          task_args.device_handle,
          ctx.get_current_global_device_id());

  auto map_instance_to_accessor = [&](DynamicValueAttrs const &value) {
    DynamicValueAttrs result = value;

    auto const &[inst, event] =
        task_args.tensor_backing.backing.at(value);

    result.accessor = dynamic_tensor_accessor_from_instance(
        inst,
        event,
        assert_unwrap(value.parallel_tensor_shape),
        Permissions::RW,
        ctx.get_current_processor());

    return result;
  };

  DynamicNodeInvocation invocation = task_args.invocation;

  invocation.inputs =
      map_values(invocation.inputs, map_instance_to_accessor);

  invocation.outputs =
      map_values(invocation.outputs, map_instance_to_accessor);

  TrainingOperationAttrs const &training_op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  PCGOperatorAttrs const &pcg_op_attrs =
      training_op_attrs.require_pcg_op();

  ReplicateAttrs const &replicate_attrs =
      pcg_op_attrs.require_parallel_replicate();

  (void)device_handle;
  (void)replicate_attrs;

  int nccl_version = 0;
  ncclResult_t result = ncclGetVersion(&nccl_version);

  if (result != ncclSuccess) {
    std::printf("NCCL error: %s\n", ncclGetErrorString(result));
    return;
  }

  std::printf("NCCL version: %d\n", nccl_version);
}

Realm::Event spawn_nccl_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    TensorInstanceBacking const &tensor_backing,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &device_handle,
    Realm::Event precondition) {
  NCCLTaskArgs task_args = NCCLTaskArgs{
      /*invocation=*/invocation,
      /*tensor_backing=*/tensor_backing,
      /*device_handle=*/device_handle,
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
