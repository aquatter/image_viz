#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/subscription_options.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <shared_mutex>
#include <thread>

class FPSMeter {
public:
  FPSMeter(size_t window) : wnd_{window} {
    std::chrono::system_clock::now().time_since_epoch().count();
  }

  void tick() {

    std::unique_lock lock{m_};
    time_vec_.push_back(
        std::chrono::system_clock::now().time_since_epoch().count());

    if (time_vec_.size() > wnd_) {
      time_vec_.pop_front();
    }
  }

  [[nodiscard]] std::optional<float> fps() {

    std::shared_lock lock{m_};

    if (time_vec_.size() < wnd_) {
      return std::nullopt;
    }

    float diff_sum{0.0f};

    for (auto i{0ul}; i < wnd_ - 1; ++i) {
      diff_sum += static_cast<float>(time_vec_[i + 1] - time_vec_[i]);
    }

    diff_sum /= static_cast<float>(wnd_ - 1);

    return 1.0e9f / diff_sum;
  }

private:
  size_t wnd_;
  std::deque<uint64_t> time_vec_;
  std::shared_mutex m_;
};

class ImageViz : public rclcpp::Node {
public:
  ImageViz() : rclcpp::Node{"cpp_img_viz"} {

    img_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "/camera/image_raw/compressed", 10,
        [this](sensor_msgs::msg::CompressedImage::UniquePtr msg) {
          img_fps_.tick();

          std::lock_guard lock{img_protector_};
          img_queue_.push_back(std::move(msg));

          if (img_queue_.size() > 2) {
            img_queue_.pop_front();
          }
        });

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/mpu6050", 10,
        [this](sensor_msgs::msg::Imu::UniquePtr) { imu_fps_.tick(); });

    gps_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        "/fix", 10,
        [this](sensor_msgs::msg::NavSatFix::UniquePtr) { gps_fps_.tick(); });

    wt_ = std::make_unique<std::thread>([this]() {
      cv::namedWindow("debug", cv::WINDOW_FULLSCREEN);
      cv::setWindowProperty("debug", cv::WND_PROP_FULLSCREEN,
                            cv::WINDOW_FULLSCREEN);
      while (not stop_requested_) {

        bool queue_empty{true};
        {
          std::lock_guard lock{img_protector_};
          queue_empty = img_queue_.empty();
        }

        if (queue_empty) {
          std::this_thread::yield();
          continue;
        }

        sensor_msgs::msg::CompressedImage::UniquePtr msg;
        {
          std::lock_guard lock{img_protector_};
          msg = std::move(img_queue_.front());
          img_queue_.pop_front();
        }

        cv::Mat_<cv::Vec3b> img =
            cv::imdecode({msg->data.data(), static_cast<int>(msg->data.size())},
                         cv::IMREAD_UNCHANGED);

        cv::resize(img, img, {640, 480});

        if (auto fps{img_fps_.fps()}; fps.has_value()) {
          cv::putText(img, fmt::format("camera: {:.1f}", fps.value()), {10, 20},
                      cv::FONT_HERSHEY_COMPLEX, 0.6, {0.0, 255.0, 0.0}, 1,
                      cv::LINE_AA);
        }

        if (auto fps{imu_fps_.fps()}; fps.has_value()) {
          cv::putText(img, fmt::format("imu: {:.1f}", fps.value()), {10, 40},
                      cv::FONT_HERSHEY_COMPLEX, 0.6, {0.0, 255.0, 0.0}, 1,
                      cv::LINE_AA);
        }

        if (auto fps{gps_fps_.fps()}; fps.has_value()) {
          cv::putText(img, fmt::format("gps: {:.1f}", fps.value()), {10, 60},
                      cv::FONT_HERSHEY_COMPLEX, 0.6, {0.0, 255.0, 0.0}, 1,
                      cv::LINE_AA);
        }

        cv::imshow("debug", img);

        if (cv::waitKey(1) == 27) {
        }
      }

      cv::destroyAllWindows();
    });
  }

  ~ImageViz() {
    if (wt_->joinable()) {
      stop_requested_ = true;
      wt_->join();
    }
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;

  FPSMeter img_fps_{10};
  FPSMeter imu_fps_{100};
  FPSMeter gps_fps_{10};

  std::unique_ptr<std::thread> wt_;
  std::deque<sensor_msgs::msg::CompressedImage::UniquePtr> img_queue_;
  std::mutex img_protector_;
  std::atomic_bool stop_requested_{false};
};

int main(const int argc, const char *const *argv) {

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViz>());
  rclcpp::shutdown();

  return EXIT_SUCCESS;
}
