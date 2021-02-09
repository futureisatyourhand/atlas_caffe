
#include <vector>
 
#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"
 
namespace caffe {
 
//真正实现置换的函数
template <typename Dtype>
void Permute(const int count, Dtype* bottom_data, const bool forward,
    const int* permute_order, const int* old_steps, const int* new_steps,
    const int num_axes, Dtype* top_data) {
    for (int i = 0; i < count; ++i) {
      int old_idx = 0;
      int idx = i;
      for (int j = 0; j < num_axes; ++j) {
        int order = permute_order[j];
        old_idx += (idx / new_steps[j]) * old_steps[order]; //old_idx为原始数据对应于现在的i的索引
        idx %= new_steps[j];
      }
      if (forward) {
        top_data[i] = bottom_data[old_idx];
      } else {
        bottom_data[old_idx] = top_data[i];
      }
    }
}
 
//PermuteLayer建立，并初始化一些参数
template <typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PermuteParameter permute_param = this->layer_param_.permute_param();
  CHECK_EQ(bottom.size(), 1);
  num_axes_ = bottom[0]->num_axes(); //获取输入blob的轴数目
  vector<int> orders;
  // Push the specified new orders.
  //将指定的新的索引轴顺序压入orders
  for (int i = 0; i < permute_param.order_size(); ++i) {
    int order = permute_param.order(i);
    CHECK_LT(order, num_axes_)
        << "order should be less than the input dimension.";
    //find()函数可参见https://www.cnblogs.com/chinshing/p/3984333.html
    if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
      LOG(FATAL) << "there are duplicate orders";
    }
    orders.push_back(order);
  }
  // Push the rest orders. And save original step sizes for each axis.
  //注意所指定的新的索引轴顺序的大小不一定等于num_axes_,例如原来顺序为0,1,2,3;指定前两轴交换顺序，即交换后为1,0,2,3
  //这时只指定permute_param.order(0)=1,permute_param.order(1)=0即可，也即只需要permute_param.order_size()=2,后两轴无需指定
  //通过以下for循环自动设置
  for (int i = 0; i < num_axes_; ++i) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  CHECK_EQ(num_axes_, orders.size());
  // Check if we need to reorder the data or keep it.检查是否需要改变数据的索引轴顺序
  need_permute_ = false;
  for (int i = 0; i < num_axes_; ++i) {
    if (orders[i] != i) {
      // As long as there is one order which is different from the natural order
      // of the data, we need to permute. Otherwise, we share the data and diff.
      //只要有一个轴的顺序发生改变，则需要置换顺序（即设置need_permute_为true）
      need_permute_ = true;
      break;
    }
  }
 
  vector<int> top_shape(num_axes_, 1);  //用于记录置换顺序后的输出blob的大小
  //以下三个变量均为blob类，方便.cu文件的实现
  permute_order_.Reshape(num_axes_, 1, 1, 1); //用于记录置换顺序后的各轴顺序
  old_steps_.Reshape(num_axes_, 1, 1, 1);
  new_steps_.Reshape(num_axes_, 1, 1, 1);
  for (int i = 0; i < num_axes_; ++i) {
    permute_order_.mutable_cpu_data()[i] = orders[i];  //将置换顺序写入permute_order_（blob）中
    top_shape[i] = bottom[0]->shape(orders[i]);  //将置换顺序后的输出blob的大小依次写入top_shape中
  }
  top[0]->Reshape(top_shape); //根据top_shape重新修正输出blob的大小
}
 
 
template <typename Dtype>
void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps_.mutable_cpu_data()[i] = 1;
    } else {
      old_steps_.mutable_cpu_data()[i] = bottom[0]->count(i + 1); //count(int start_axis)实现计算从某一维度开始的元素总数
    }
    top_shape.push_back(bottom[0]->shape(permute_order_.cpu_data()[i]));
  }
  top[0]->Reshape(top_shape); //感觉多此一举（上面建立层的函数已经reshape过了）
  
  
  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      new_steps_.mutable_cpu_data()[i] = 1;
    } else {
      new_steps_.mutable_cpu_data()[i] = top[0]->count(i + 1);
    }
  }
}
 
//前向传播
template <typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (need_permute_) {
    Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int top_count = top[0]->count();
    const int* permute_order = permute_order_.cpu_data();
    const int* old_steps = old_steps_.cpu_data();
    const int* new_steps = new_steps_.cpu_data();
    bool forward = true; 
    //调用Permute()函数实现输入数据的索引轴顺序置换
    Permute(top_count, bottom_data, forward, permute_order, old_steps,
            new_steps, num_axes_, top_data);
  } else {
    // If there is no need to permute, we share data to save memory.
    top[0]->ShareData(*bottom[0]); //输出共享输入数据，节省内存
  }
}
 
//后向传播（其实就是将输出diff改回原顺序赋值给输入diff，从而实现后向传播）
template <typename Dtype>
void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (need_permute_) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int top_count = top[0]->count();
    const int* permute_order = permute_order_.cpu_data();
    const int* old_steps = old_steps_.cpu_data();
    const int* new_steps = new_steps_.cpu_data();
    bool forward = false;
    Permute(top_count, bottom_diff, forward, permute_order, old_steps,
            new_steps, num_axes_, top_diff);
  } else {
    // If there is no need to permute, we share diff to save memory.
    bottom[0]->ShareDiff(*top[0]);
  }
}
 
#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
#endif
 
INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);
 
} 
