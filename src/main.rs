#![feature(iterator_fold_self)]
#![allow(dead_code, unused_imports)]
use image::{ImageBuffer, Rgb};
use imageproc::contours::{find_contours_with_threshold, Contour};
use imageproc::contrast::threshold;
use imageproc::edges::canny;
use imageproc::point::Point;
use std::f64::consts::{E, PI};
use plotters::prelude::*;

fn main() -> Result<(), anyhow::Error> {
    let areas = analyze_sample("sample-1")?;

    let root =
        BitMapBackend::new("sample-1-hist.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .build_cartesian_2d(0u32..2000u32, 0u32..40u32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("# of Particles")
        .x_desc("Area in Pixels")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.filled())
            .data(areas.iter().map(|x: &f64| (*x as u32, 2))),
    )?;

    Ok(())
}

fn analyze_sample(name: &str) -> Result<Vec<f64>, anyhow::Error> {

    let sample = image::open(format!("{}.png", name))?;

    let l8_img = sample.to_luma8();

    let threshold_num = 170;

    let mut contours: Vec<Contour<usize>> = find_contours_with_threshold(&l8_img, threshold_num);
    contours.remove(0);

    let mut rgb_img = sample.to_rgb8();

    let mut areas = vec![];
    for c in contours.into_iter() {
        // println!("{}", contour_area(&c));
        let contour_area = greens_theorem(&c);
        areas.push(contour_area);

        let max_i = c.points.len();
        for (i, p) in c.points.iter().enumerate() {
            let variance = (255. * (i as f64 / max_i as f64)) as u8;
            match c.border_type {
                imageproc::contours::BorderType::Outer => {
                    rgb_img.put_pixel(p.x as u32, p.y as u32, Rgb([255, variance, 0]))
                }
                imageproc::contours::BorderType::Hole => {
                    rgb_img.put_pixel(p.x as u32, p.y as u32, Rgb([variance, 255, 0]))
                }
            }
        }
    }

    rgb_img.save(format!("{}-contours.png", name))?;
    Ok(areas)
}

fn greens_theorem(contour: &Contour<usize>) -> f64 {
    let p = contour.points.iter();
    let mut p_shift = contour.points.iter().cycle();
    p_shift.next();

    p.zip(p_shift)
        .map(|(p0, p1)| {
            (
                Point::new(p0.x as f64, p0.y as f64),
                Point::new(p1.x as f64, p1.y as f64),
            )
        })
        .fold(0., |sum, (p0, p1)| sum + (p0.x * p1.y - p1.x * p0.y))
        .abs() / 2.
}

fn create_norm_hist_svg(mu: f64, sigma: f64) {
    let domain = generate_domain(mu - 4. * sigma, mu + 4. * sigma, 0.1);
    let _sample = norm(mu, sigma, &domain);
}

fn generate_domain(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let shift = 1.0 / step;
    let a = (start * shift) as isize;
    let b = (stop * shift) as isize;

    (a..b).into_iter().map(|x| x as f64 / shift).collect()
}

fn norm(mu: f64, sigma: f64, domain: &Vec<f64>) -> Vec<f64> {
    domain
        .iter()
        .map(|x| {
            let count = (norm_pdf(x, mu, sigma) * domain.len() as f64) as usize;
            vec![x.clone(); count]
        })
        // Probability Density function of a normal distribution
        .flatten()
        .collect()
}

fn norm_pdf(x: &f64, mu: f64, sigma: f64) -> f64 {
    (1.0 / (sigma * (2.0 * PI).sqrt())) * E.powf(-0.5 * ((x - mu) / sigma).powi(2))
}
