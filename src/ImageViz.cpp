
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fmt/color.h>
#include <fmt/format.h>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/client.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/subscription_options.hpp>
#include <rclcpp/utilities.hpp>
#include <rosbag2_interfaces/srv/pause.hpp>
#include <rosbag2_interfaces/srv/resume.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <shared_mutex>
#include <string_view>
#include <thread>

using namespace std::chrono_literals;

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

  [[nodiscard]] std::optional<float> fps() const {
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
  mutable std::shared_mutex m_;
};

class ImageViz : public rclcpp::Node {
public:
  ImageViz() : rclcpp::Node{"cpp_img_viz"} {

    img_size_.width = 640;
    img_size_.height = 400;

    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        [this](sensor_msgs::msg::Image::UniquePtr msg) {
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
        "/fix", 10, [this](sensor_msgs::msg::NavSatFix::UniquePtr msg) {
          gps_is_valid_ = std::isnan(msg->latitude) or
                                  std::isnan(msg->longitude) or
                                  std::isnan(msg->altitude)
                              ? false
                              : true;
          gps_fps_.tick();
        });

    resume_ = create_client<rosbag2_interfaces::srv::Resume>(
        "/rosbag2_recorder/resume");
    pause_ = create_client<rosbag2_interfaces::srv::Pause>(
        "/rosbag2_recorder/pause");

    wt_ = std::make_unique<std::thread>([this]() {
      cv::namedWindow("debug", cv::WINDOW_FULLSCREEN);
      cv::setWindowProperty("debug", cv::WND_PROP_FULLSCREEN,
                            cv::WINDOW_FULLSCREEN);

      auto record_start_time{std::chrono::system_clock::now()};
      auto last_blink_time{std::chrono::system_clock::now()};
      bool blink_on{false};

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

        sensor_msgs::msg::Image::UniquePtr msg;
        {
          std::lock_guard lock{img_protector_};
          msg = std::move(img_queue_.front());
          img_queue_.pop_front();
        }

        cv::Mat_<cv::Vec3b> img =
            cv::Mat_<uint8_t>{msg->data, true}.reshape(3, msg->height);

        if (focusing_mode_) {
          img = img(
              cv::Rect((img.cols >> 1) - 100, (img.rows >> 1) - 100, 200, 200));

          cv::Mat_<uint8_t> img8b;
          cv::cvtColor(img, img8b, cv::COLOR_BGR2GRAY);

          cv::Mat_<float> laplacian_img;
          cv::Laplacian(img8b, laplacian_img, CV_32FC1, 5);
          const auto sharpness_metric{cv::mean(cv::abs(laplacian_img))};

          draw_text(img, fmt::format("{:.1f}", sharpness_metric(0)), {10, 25},
                    {0.0, 0.0, 255.0});
        } else {
          cv::resize(img, img, img_size_, 0.0, 0.0, cv::INTER_NEAREST);
          cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
          draw(img);

          if (record_started_) {
            const auto current_time{std::chrono::system_clock::now()};

            if ((current_time - last_blink_time) > 100ms) {
              last_blink_time = current_time;
              blink_on = not blink_on;

              if (blink_on) {
                cv::circle(img, {50, 350}, 20, {0.0, 0.0, 255.0}, cv::FILLED,
                           cv::LINE_AA);
              }
            }

            const auto record_time{
                std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - record_start_time)
                    .count()};

            draw_text(img, fmt::format("{} s", record_time), {80, 355},
                      {0.0, 255.0, 0.0});
          }
        }

        cv::imshow("debug", img);

        const auto key_pressed{cv::waitKey(1)};

        if (key_pressed == ' ' and not focusing_mode_) {

          if (not record_started_) {

            auto res{resume_->async_send_request(
                std::make_shared<rosbag2_interfaces::srv::Resume::Request>())};

            if (res.wait_for(5s) == std::future_status::ready) {
              RCLCPP_INFO(get_logger(), "%s",
                          fmt::format(fmt::fg(fmt::color::green_yellow),
                                      "Recording started..")
                              .c_str());

              record_started_ = true;
              record_start_time = std::chrono::system_clock::now();
            }

          } else {

            auto res{pause_->async_send_request(
                std::make_shared<rosbag2_interfaces::srv::Pause::Request>())};

            if (res.wait_for(5s) == std::future_status::ready) {
              RCLCPP_INFO(get_logger(), "%s",
                          fmt::format(fmt::fg(fmt::color::green_yellow),
                                      "Recording stopped..")
                              .c_str());

              record_started_ = false;
            }
          }
        }

        if (key_pressed == 'f' and not record_started_) {
          focusing_mode_ = not focusing_mode_;
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
  void draw(cv::Mat_<cv::Vec3b> img) const {
    int y_value{25};

    if (auto fps{img_fps_.fps()}; fps.has_value()) {
      draw_text(img, fmt::format("camera: {:.1f}", fps.value()), {10, y_value},
                {0.0, 255.0, 0.0});
    }

    if (auto fps{imu_fps_.fps()}; fps.has_value()) {
      y_value += 30;
      draw_text(img, fmt::format("imu: {:.1f}", fps.value()), {10, y_value},
                {0.0, 255.0, 0.0});
    }

    if (auto fps{gps_fps_.fps()}; fps.has_value()) {
      y_value += 30;
      const auto clr{gps_is_valid_ ? cv::Scalar{0.0, 255.0, 0.0}
                                   : cv::Scalar{0.0, 0.0, 255.0}};
      draw_text(img, fmt::format("gps: {:.1f}", fps.value()), {10, y_value},
                clr);
    }
  }

  void draw_text(cv::Mat_<cv::Vec3b> img, const std::string_view msg,
                 const cv::Point p, const cv::Scalar clr) const {
    cv::putText(img, msg.data(), p, cv::FONT_HERSHEY_COMPLEX, 0.6,
                cv::Scalar::all(0.0), 7, cv::LINE_AA);

    cv::putText(img, msg.data(), p, cv::FONT_HERSHEY_COMPLEX, 0.6, clr, 1,
                cv::LINE_AA);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Client<rosbag2_interfaces::srv::Resume>::SharedPtr resume_;
  rclcpp::Client<rosbag2_interfaces::srv::Pause>::SharedPtr pause_;

  FPSMeter img_fps_{10};
  FPSMeter imu_fps_{100};
  FPSMeter gps_fps_{50};

  std::unique_ptr<std::thread> wt_;
  std::deque<sensor_msgs::msg::Image::UniquePtr> img_queue_;
  std::mutex img_protector_;
  std::atomic_bool stop_requested_{false};
  std::atomic_bool gps_is_valid_{false};
  std::atomic_bool focusing_mode_{false};
  std::atomic_bool record_started_{false};

  cv::Size img_size_;
};

int main(const int argc, const char *const *argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViz>());
  rclcpp::shutdown();

  return EXIT_SUCCESS;
}
