#include "realm-execution/tasks/impl/serializable_nccl_task_args.h"
#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"

namespace FlexFlow {

SerializableNcclTaskArgs
nccl_task_args_to_serializable(NcclTaskArgs const &args) {
  return SerializableNcclTaskArgs{
      /*invocation=*/
      dynamic_node_invocation_to_serializable(args.invocation),
      /*tensor_backing=*/
      tensor_instance_backing_to_serializable(args.tensor_backing),
  };
}

NcclTaskArgs
nccl_task_args_from_serializable(SerializableNcclTaskArgs const &args) {
  return NcclTaskArgs{
      /*invocation=*/
      dynamic_node_invocation_from_serializable(args.invocation),
      /*tensor_backing=*/
      tensor_instance_backing_from_serializable(args.tensor_backing),
  };
}

} // namespace FlexFlow
