// Copyright [2023] <Mhammad Rumi>
#include "coco.hpp"  // NOLINT

#include <algorithm>

coco::coco(const std::string& file) {
  validation_file = file;
  PRINT("Loading validation Truth form file: ", validation_file);
  create_index();
}
void coco::create_index() {
  std::cout << "Creating Index... " << std::endl;
  std::fstream file(validation_file);
  std::cout << "Loading annnotations to memory..." << std::endl;
  json dataset = json::parse(file);

  std::vector<float> bbox;
  int id, category_id, image_id, iscrowd;
  float score, area;
  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      EXTRACT(image_id, ann);
      EXTRACT(iscrowd, ann);
      EXTRACT(id, ann);
      EXTRACT(area, ann);
      EXTRACT(category_id, ann);
      EXTRACT(bbox, ann);
      gt[{image_id, category_id}].push_back(
          {id, iscrowd, image_id, category_id, 1., area, bbox});
      catToImgs[category_id].push_back(image_id);
    }
  }
  if (dataset.contains("images")) {
    for (auto&& img : dataset["images"]) {
      imgIds.push_back(img["id"]);
    }
  }
  if (dataset.contains("categories")) {
    for (auto&& cat : dataset["categories"]) {
      catIds.push_back(cat["id"]);
    }
  }
  // Print the size of the maps.
  PRINT("Number of images lable pair: ", gt.size());
  PRINT("Number of categories: ", catIds.size());
  PRINT("Total no. of images: ", imgIds.size());
  SEPARATOR;
}
void coco::loadRes(const std::string resFile) {
  std::fstream file(resFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file:" + resFile);
  }

  detection_file = resFile;
  PRINT("Loading results from: ", detection_file);
  json result = json::parse(file);

  std::vector<float> bbox;
  int id = 1, category_id, image_id;
  float score, area;
  std::cout << "Processing... " << std::endl;
  for (const auto& ann : result) {
    EXTRACT(image_id, ann);
    EXTRACT(score, ann);
    EXTRACT(category_id, ann);
    EXTRACT(bbox, ann);
    area = bbox[3] * bbox[2];
    dt[{image_id, category_id}].push_back(
        {id, 0, image_id, category_id, score, area, bbox});
    id++;
  }
  PRINT("Number of images lable pair: ", dt.size());
  SEPARATOR;
}
float coco::iou(const std::vector<float>& gt_bbox,
                const std::vector<float>& dt_bbox, const int& crowd) {
  /*
  input:
  gt_bbox = [xmin, ymin, w, h]
  dt_bbox = [xmin, ymin, w, h]
  ouput:
  float iou;
  */
  // PRINT("gt_bbox: ", gt_bbox.size());
  // PRINT("dt_bbox: ", dt_bbox.size());
  // return 0;
  assert(gt_bbox.size() == 4 && dt_bbox.size() == 4);
  float x1, y1, x2, y2;
  x1 = std::max(gt_bbox[0], dt_bbox[0]);
  y1 = std::max(gt_bbox[1], dt_bbox[1]);
  x2 = std::min(gt_bbox[0] + gt_bbox[2], dt_bbox[0] + dt_bbox[2]);
  y2 = std::min(gt_bbox[1] + gt_bbox[3], dt_bbox[1] + dt_bbox[3]);
  if (x2 < x1 || y2 < y1) return 0.0;  // if there is no overlap between bboxes

  float i = (x2 - x1) * (y2 - y1);
  float area1 = (gt_bbox[2]) * (gt_bbox[3]);
  float area2 = (dt_bbox[2]) * (dt_bbox[3]);
  float u = crowd ? area2 : (area1 + area2 - i);
  return i / u;
}
match coco::evalImg(const int& imgId, const int& catId,
                    const point<float>& aRng, const int& maxDet) {
  auto t = val_params.iou_Thrs;
  auto& g_img_cat = gt[{imgId, catId}];
  auto& d_img_cat = dt[{imgId, catId}];
  auto ious_sel = ious[{imgId, catId}];
  std::vector<float> score;

  int T = t.size(), D = d_img_cat.size(), G = g_img_cat.size();
  std::vector<std::vector<float>> dtm(T, std::vector<float>(D, 0.f)),
      dtIg(T, std::vector<float>(D, 0.f));
  std::vector<std::vector<float>> gtm(T, std::vector<float>(G, 0.f));
  std::vector<bool> gtIg;
  std::transform(g_img_cat.begin(), g_img_cat.end(), std::back_inserter(gtIg),
                 [aRng](label& value) {
                   static int i = 0;
                   return aRng(value.area) and value.is_crowd == 1;  // NOLINT
                   i++;
                 });
  //  sort dt highest score first, sort gt ignore last
  std::sort(gtIg.begin(), gtIg.end());
  std::sort(d_img_cat.begin(), d_img_cat.end(),
            [](label& x, label& y) { return x.scores > y.scores; });
  if (!ious_sel.size() == 0) {
    for (int tind = 0; tind < t.size(); ++tind) {
      for (int dind = 0; dind < d_img_cat.size(); ++dind) {
        int m = -1;
        float iou = std::min(t[tind], static_cast<float>(EP0));
        score.push_back(d_img_cat[dind].scores);
        for (int gind = 0; gind < g_img_cat.size(); ++gind) {
          //  if this gt already matched, and not a crowd, continue
          if (gtm[tind][gind] > 0 && (!g_img_cat[gind].is_crowd)) continue;
          // if dt matched to reg gt, and on ignore gt, stop
          if (m > -1 && gtIg[m] == 0 && gtIg[gind] == 1) break;
          // continue to next gt unless better match made
          if (ious_sel[dind][gind] < iou) continue;
          // if match successful and best so far, store appropriately
          iou = ious_sel[dind][gind];
          m = gind;
        }
        // if match made store id of match for both dt and gt
        if (m == -1) continue;
        dtm[tind][dind] = g_img_cat[m].id;
        gtm[tind][m] = d_img_cat[dind].id;
        dtIg[tind][dind] = gtIg[m];
      }
    }
  }
  return {imgId, catId, maxDet, aRng, dtm, gtm, dtIg, score, gtIg};
}
void coco::get_scores() {
  PRINT("Calculating IOUs...", "");
  for (auto&& [key, dt_ann] : dt) {
    _map_label::iterator g_ann = gt.find(key);
    if (g_ann == gt.end()) {
      coco::ious[key].push_back({});
      continue;
    }
    std::sort(dt_ann.begin(), dt_ann.end(),
              [](label x, label y) { return x.scores > y.scores; });
    for (auto&& d : dt_ann) {
      std::vector<float> temp;
      for (auto&& g : g_ann->second) {
        int catId = g.catids;
        auto a = g.bbox;
        // if (g.is_crowd == true) continue;  // catId != d.catids &&
        float temps = coco::iou(g.bbox, d.bbox, g.is_crowd);
        temp.push_back(temps);
      }
      coco::ious[key].push_back(temp);
      // std::for_each(temp.begin(), temp.end(),
      //               [](const float& i) { std::cout << i << ", "; });
      // std::cout << std::endl;
    }
  }
  PRINT("IOUs calculated: ", ious.size());
  SEPARATOR;
}
// std::map<int, coco::_curve> coco::precision_recall(
//     const std::vector<float>& thres) {
//   std::cout << "Calculating precision recall" << std::endl;

//   std::map<int, _curve> pr;
//   std::map<int, std::vector<point<int>>> per_sample_tp_dt;
//   std::vector<int> truePos(91), total_gt(91), total_dt(91);

//   for (const auto& [catId, imgIds] : catToImgs) {
//     int TP = 0, gt_percat = 0, dt_percat = 0;
//     for (const auto& imgId : imgIds) {
//       Key<int> keys = {imgId, catId};
//       const auto& current_per_img_cat_id_iou = ious[keys];
//       PRINT("current key: ", keys);
//       PRINT("Size of IOU: ", current_per_img_cat_id_iou.size());
//       auto x = count_x(gt[keys], catId);
//       PRINT("total cases in ground truth: ", x);
//       // std::for_each(current_per_img_cat_id_iou.begin(),
//       //               current_per_img_cat_id_iou.end(),
//       //               [](const float& j) { std::cout << j << ", "; });

//       // getting True positives upon IOU threshold.

//       std::vector<int> matches;
//       // PRINT("Size of ")
//       std::transform(current_per_img_cat_id_iou.begin(),
//                      current_per_img_cat_id_iou.end(),
//                      std::back_inserter(matches), [thres](float iou) {
//                        return iou > 0.5;  // left to iterate over the
//                        thresholds
//                      });
//       // PRINT("Matches size: ", matches.size());
//       auto tp = std::accumulate(matches.begin(), matches.end(), 0);
//       // if (tp > x) tp = x;
//       // TP += tp;
//       PRINT("Total True Positive: ", tp);
//       assert(tp <= x);
//       dt_percat += matches.size();

//       per_sample_tp_dt[catId].push_back({TP, dt_percat});
//       // total samples of a category in the entire dataset
//       gt_percat += x;
//     }
//     // SEPARATOR;
//     truePos[catId] = TP;
//     total_dt[catId] = dt_percat;
//     total_gt[catId] = gt_percat;
//     // PRINT("Catid: ", catId);
//     // SEPARATOR;
//     // PRINT("TP: ", TP);
//     // PRINT("TP+FN: ", gt_percat);
//     // SEPARATOR;
//     // PRINT("total ground truth ann per category", total_gt[catId]);
//   }
//   float P, R;
//   for (auto&& [cats, tp_dt_per_cat] : per_sample_tp_dt) {
//     for (int i = 0; i < tp_dt_per_cat.size(); ++i) {
//       P = static_cast<float>(tp_dt_per_cat[i].x) /
//           (static_cast<float>(tp_dt_per_cat[i].y) + EP0);
//       R = static_cast<float>(tp_dt_per_cat[i].x) /
//           (static_cast<float>(total_gt[cats]) + EP0);
//       pr[cats].push_back({P, R});
//     }
//   }

//   // for_each(pr.begin(), pr.end(),
//   //          [](std::pair<const int, std::vector<point<float>>> a) {
//   //            std::cout << a.first << ": ";
//   //            for_each(a.second.begin(), a.second.end(),
//   //                     [](point<float> j) { std::cout << j; });
//   //            SEPARATOR;
//   //          });
//   count_x(truePos, 0);
//   PRINT("len of pr variable: ", pr.size());
//   return pr;
// }

void coco::evaluation(const float* IOU_range) {
  // compute IOUs.
  get_scores();
  // auto test = ious.find({139, 62});
  // std::for_each(ious.begin(), ious.end(),
  //               [](std::pair<Key, std::vector<float>> a) {
  //                 Key x = {440475, 1};
  //                 if (a.first == x) {
  //                   PRINT("Current Key: ", a.first);
  //                   std::for_each(a.second.begin(), a.second.end(),
  //                                 [](float& j) { std::cout << j << ", ";
  //                                 });
  //                   std::cout << std::endl;
  //                   SEPARATOR;

  //                 } else {
  //                   auto x = 1;
  //                 }
  //               });
  PRINT("Matching pairs...", "");
  for (auto&& catId : catIds) {
    for (auto&& areaRng : val_params.aRng) {
      for (auto&& imgId : imgIds) {
        mapped.push_back(evalImg(imgId, catId, areaRng, 100));
      }
    }
  }
  PRINT("completed mapping", "");

  // auto preci_recall_vals = precision_recall(iouThrs);
  // for (int i = 0; i < 91; i++) {
  //   std::ofstream file("logs/preci_recall_" + std::to_string(i) + ".txt");
  //   if (!file.is_open()) {
  //     std::cerr << "Error opening the file!"
  //               << std::endl;  // Return an error code
  //   }

  //   for (auto&& i : preci_recall_vals[i]) {
  //     file << i << std::endl;
  //   }
  //   file.close();
  // }
  // }
}
void coco::accumulate() {
  int T = val_params.iou_Thrs.size(), R = val_params.recThrs.size(),
      K = catIds.size(), A = val_params.aRng.size(),
      M = val_params.maxDet.size();

  tensor precision({T, R, K, A, M}, -1);
  tensor recall({T, K, A, M}, -1);
  tensor scores({T, R, K, A, M}, -1);

  for (int i = 0; i < T; ++i) {
    for (int a = 0; a < A; a++) {
      for (int m = 0; m < M; m++) {
        // do stuff here
      }
    }
  }
}
coco::~coco() {}
// goal
//  yolov5 ko intergrate karna hoga n m x intergrate karna generic code  look for other code model as refrence.
