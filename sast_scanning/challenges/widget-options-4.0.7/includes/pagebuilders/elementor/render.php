<?php

/**
 * Extends funtionality to Elementor Pagebuilder
 *
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       4.3
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;

if (!function_exists('widgetopts_elementor_render')) {
	add_action('elementor/widget/render_content', 'widgetopts_elementor_render', 10, 2);
	function widgetopts_elementor_render($content, $widget)
	{
		if (!Elementor\Plugin::$instance->editor->is_edit_mode()) {
			global $widget_options;
			$settings 	= $widget->get_settings();

			$hidden     		= false;
			$placeholder 		= '<div class="widgetopts-placeholder-e"></div>';
			$visibility_opts    = isset($settings['widgetopts_visibility']) ? $settings['widgetopts_visibility'] : 'hide';

			$tax_opts   = (isset($widget_options['settings']) && isset($widget_options['settings']['taxonomies_keys'])) ? $widget_options['settings']['taxonomies_keys'] : array();
			$is_misc    = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['misc'])) ? true : false;
			$is_types   = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['post_type'])) ? true : false;
			$is_tax     = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['taxonomies'])) ? true : false;

			//pages
			if ($is_misc && ((is_home() && is_front_page()) || is_front_page())) {
				if (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && in_array('home', $settings['widgetopts_misc']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings['widgetopts_misc']) || (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && !in_array('home', $settings['widgetopts_misc']))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_home', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_misc && is_home()) {
				if (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && in_array('blog', $settings['widgetopts_misc']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings['widgetopts_misc']) || (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && !in_array('blog', $settings['widgetopts_misc']))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_blog', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_tax && is_category()) {
				//category page
				if (!isset($settings['widgetopts_tax_category'])) {
					$settings['widgetopts_tax_category'] = array();
				}
				if (!isset($settings['widgetopts_taxonomies'])) {
					$settings['widgetopts_taxonomies'] = array();
				}

				$cat_lists = $settings['widgetopts_tax_category'];
				if (!in_array('category', $settings['widgetopts_taxonomies']) && $visibility_opts == 'hide' && in_array(get_query_var('cat'), $cat_lists)) {
					$hidden = true; //hide if exists on hidden pages
				} elseif (!in_array('category', $settings['widgetopts_taxonomies']) && $visibility_opts == 'show' && !in_array(get_query_var('cat'), $cat_lists)) {
					$hidden = true; //hide if doesn't exists on visible pages
				} elseif (in_array('category', $settings['widgetopts_taxonomies']) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all categories
				} elseif (in_array('category', $settings['widgetopts_taxonomies']) && $visibility_opts == 'show') {
					$hidden = false; //hide to all categories
				}
				//
				// //do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_categories', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_tax && is_tag()) {
				if (!isset($settings['widgetopts_tax_post_tag'])) {
					$settings['widgetopts_tax_post_tag'] = array();
				}
				if (!isset($settings['widgetopts_taxonomies'])) {
					$settings['widgetopts_taxonomies'] = array();
				}

				if (in_array('post_tag', $settings['widgetopts_taxonomies']) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all tags
				} elseif (in_array('post_tag', $settings['widgetopts_taxonomies']) && $visibility_opts == 'show') {
					$hidden = false; //hide to all tags
				}
				//
				// //do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_tags', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_tax && is_tax()) {
				$term = get_queried_object();

				//taxonomies page
				if (!isset($settings['widgetopts_taxonomies'])) {
					$settings['widgetopts_taxonomies'] = array();
				}
				// print_r( $term_lists );
				if (in_array($term->taxonomy, $settings['widgetopts_taxonomies']) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all tags
				} elseif (!in_array($term->taxonomy, $settings['widgetopts_taxonomies']) && $visibility_opts == 'show') {
					$hidden = true; //hide to all tags
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_taxonomies', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_misc && is_archive()) {
				//archives page
				if (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && in_array('archives', $settings['widgetopts_misc']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings['widgetopts_misc']) || (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && !in_array('archives', $settings['widgetopts_misc']))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_archives', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_misc && is_404()) {
				//404 page
				if (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && in_array('404', $settings['widgetopts_misc']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings['widgetopts_misc']) || (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && !in_array('404', $settings['widgetopts_misc']))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_404', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_misc && is_search()) {
				if (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && in_array('search', $settings['widgetopts_misc']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings['widgetopts_misc']) || (isset($settings['widgetopts_misc']) && is_array($settings['widgetopts_misc']) && !in_array('search', $settings['widgetopts_misc']))) && $visibility_opts == 'show') {
					$hidden = true;
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_search', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			} elseif (is_single() && !is_page()) {
				global $wp_query;
				$post = $wp_query->post;

				if (!isset($settings['widgetopts_types']) || ($is_types && !isset($settings['widgetopts_types'])) || (isset($settings['widgetopts_types']) && !is_array($settings['widgetopts_types']))) {
					$settings['widgetopts_types'] = array();
				}

				if ($visibility_opts == 'hide' && in_array($post->post_type, $settings['widgetopts_types'])) {
					$hidden = true; //hide if exists on hidden pages
				} elseif ($visibility_opts == 'show' && !in_array($post->post_type, $settings['widgetopts_types'])) {
					$hidden = true; //hide if doesn't exists on visible pages
				}

				// do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_single', $hidden);


				// $taxonomy_names  = get_post_taxonomies( $post->ID );
				// $array_intersect = array_intersect( $tax_opts, $taxonomy_names );

				if (!isset($settings['widgetopts_tax_category'])) {
					$settings['widgetopts_tax_category'] = array();
				}

				if (isset($settings['widgetopts_tax_category']) && !empty($settings['widgetopts_tax_category'])) {
					$cats	= wp_get_post_categories($post->ID);

					if (is_array($cats) && !empty($cats)) {
						$checked_cats   = $settings['widgetopts_tax_category'];
						$intersect      = array_intersect($cats, $checked_cats);
						if (!empty($intersect) && $visibility_opts == 'hide') {
							$hidden = true;
						} elseif (!empty($intersect) && $visibility_opts == 'show') {
							$hidden = false;
						}

						$hidden = apply_filters('widgetopts_elementor_visibility_single_category', $hidden);
					}
				}

				if ($hidden) {
					return $placeholder;
				}
			} elseif ($is_types && is_page()) {
				global $wp_query;

				$post = $wp_query->post;

				//do post type condition first
				if (isset($settings['widgetopts_types'])) {
					if ($visibility_opts == 'hide' && in_array('page', $settings['widgetopts_types'])) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !in_array('page', $settings['widgetopts_types'])) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				} else {
					// print_r( $settings['widgetopts_pages'] );
					//do per pages condition
					if (!isset($settings['widgetopts_pages'])) {
						$settings['widgetopts_pages'] = array();
					}

					if ($visibility_opts == 'hide' && in_array($post->ID, $settings['widgetopts_pages'])) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !in_array($post->ID, $settings['widgetopts_pages'])) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				}

				// //do return to bypass other conditions
				$hidden = apply_filters('widgetopts_elementor_visibility_page', $hidden);
				if ($hidden) {
					return $placeholder;
				}
			}

			//ACF
			if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) {
				if (isset($settings['widgetopts_acf_field']) && !empty($settings['widgetopts_acf_field'])) {
					$acf = get_field_object($settings['widgetopts_acf_field']);
					if ($acf && is_array($acf)) {
						$acf_visibility    = isset($settings['widgetopts_acf_visibility']) ? $settings['widgetopts_acf_visibility'] : 'hide';

						//handle repeater fields
						if (isset($acf['value'])) {
							if (is_array($acf['value'])) {
								$acf['value'] = implode(', ', array_map(function ($acf_array_value) {
									$acf_implode = '';
									if (is_array($acf_array_value)) {
										$acf_implode = implode(',', array_filter($acf_array_value));
									} else {
										$acf_implode = $acf_array_value;
									}
									return $acf_implode;
								}, $acf['value']));
							}
						}
						switch ($settings['widgetopts_acf_condition']) {
							case 'equal':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && $acf['value'] == $settings['widgetopts_acf']) {
										$hidden = false;
									} else if ('show' == $acf_visibility && $acf['value'] != $settings['widgetopts_acf']) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && $acf['value'] == $settings['widgetopts_acf']) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && $acf['value'] != $settings['widgetopts_acf']) {
										$hidden = false;
									}
								}
								break;

							case 'not_equal':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && $acf['value'] == $settings['widgetopts_acf']) {
										$hidden = true;
									} else if ('show' == $acf_visibility && $acf['value'] != $settings['widgetopts_acf']) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && $acf['value'] == $settings['widgetopts_acf']) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && $acf['value'] != $settings['widgetopts_acf']) {
										$hidden = true;
									}
								}
								break;

							case 'contains':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) !== false) {
										$hidden = false;
									} else if ('show' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) === false) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) !== false) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) === false) {
										$hidden = false;
									}
								}
								break;

							case 'not_contains':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) !== false) {
										$hidden = true;
									} else if ('show' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) === false) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) !== false) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings['widgetopts_acf']) === false) {
										$hidden = true;
									}
								}
								break;

							case 'empty':
								if ('show' == $acf_visibility && empty($acf['value'])) {
									$hidden = false;
								} else if ('show' == $acf_visibility && !empty($acf['value'])) {
									$hidden = true;
								} elseif ('hide' == $acf_visibility && empty($acf['value'])) {
									$hidden = true;
								} else if ('hide' == $acf_visibility && !empty($acf['value'])) {
									$hidden = false;
								}
								break;

							case 'not_empty':
								if ('show' == $acf_visibility && empty($acf['value'])) {
									$hidden = true;
								} else if ('show' == $acf_visibility && !empty($acf['value'])) {
									$hidden = false;
								} elseif ('hide' == $acf_visibility && empty($acf['value'])) {
									$hidden = false;
								} else if ('hide' == $acf_visibility && !empty($acf['value'])) {
									$hidden = true;
								}
								break;

							default:
								# code...
								break;
						}

						// //do return to bypass other conditions
						$hidden = apply_filters('widgetopts_elementor_visibility_acf', $hidden);
						if ($hidden) {
							return $placeholder;
						}
					}
				}
			}

			//widget logic
			if (isset($widget_options['state']) && 'activate' == $widget_options['state']) {
				if (isset($settings['widgetopts_roles_state']) && !empty($settings['widgetopts_roles_state'])) {
					//do state action here
					if ($settings['widgetopts_roles_state'] == 'out' && is_user_logged_in()) {
						return $placeholder;
					} else if ($settings['widgetopts_roles_state'] == 'in' && !is_user_logged_in()) {
						return $placeholder;
					}
				}
			}

			//widget logic
			if ('activate' == $widget_options['logic']) {
				if (isset($settings['widgetopts_logic']) && !empty($settings['widgetopts_logic'])) {
					//do widget logic
					$display_logic = stripslashes(trim($settings['widgetopts_logic']));
					$display_logic = apply_filters('widget_options_logic_override', $display_logic);
					$display_logic = apply_filters('extended_widget_options_logic_override', $display_logic);
					if ($display_logic === false) {
						return $placeholder;
					}
					if ($display_logic === true) {
						return $content;
					}
					// if (stristr($display_logic, "return") === false) {
					// 	$display_logic = "return (" . $display_logic . ");";
					// }
					$display_logic = htmlspecialchars_decode($display_logic, ENT_QUOTES);
					try {
						if (!widgetopts_safe_eval($display_logic)) {
							return $placeholder;
						}
					} catch (ParseError $e) {
						return $placeholder;
					}
				}
			}
		}

		return $content;
	}
}

if (!function_exists('widgetopts_elementor_before_render')) {
	add_action('elementor/frontend/widget/before_render', 'widgetopts_elementor_before_render', 10, 2);
	function widgetopts_elementor_before_render($element)
	{
		$enabled = array('button', 'button_plus', 'eael-creative-button', 'cta');
		if (in_array($element->get_name(), $enabled)) {
			global $widget_options;
			if ('activate' == $widget_options['sliding']) {
				$settings = $element->get_settings();
				if (isset($settings['widgetopts_open_sliding']) && 'on' == $settings['widgetopts_open_sliding']) {
					$element->add_render_attribute('button', 'class', 'sl-widgetopts-open');
				}
			}
		}
	}
}

if (!function_exists('widgetopts_elementor_extra_js')) {
	add_action('wp_footer', 'widgetopts_elementor_extra_js');
	function widgetopts_elementor_extra_js()
	{ ?>
		<script type="text/javascript">
			(function($, window, document, undefined) {
				if (jQuery('.widgetopts-placeholder-e').length > 0) {
					// jQuery('.elementor-column-wrap:has(.widgetopts-placeholder-e)').hide();

					jQuery('.elementor-section:has(.widgetopts-placeholder-e)').each(function() {
						var pTop = jQuery(this).find('.elementor-element-populated').css('padding-top');
						var pBot = jQuery(this).find('.elementor-element-populated').css('padding-bottom');
						var pHeight = jQuery(this).find('.elementor-element-populated').innerHeight();
						var vert = pHeight - (parseFloat(pTop) + parseFloat(pBot));

						if (typeof vert !== 'undefined' && vert < 5) {
							jQuery(this).hide();
						} else {
							jQuery(this).find('.widgetopts-placeholder-e').each(function() {
								jQuery(this).closest('.elementor-element').hide();

								var countEl = jQuery(this).closest('.elementor-column').find('.elementor-element').length;
								var countHolder = jQuery(this).closest('.elementor-column').find('.widgetopts-placeholder-e').length;
								if (countEl == countHolder) {
									jQuery(this).closest('.elementor-column').hide();
								}
							}).promise().done(function() {
								var sTop = jQuery(this).closest('.elementor-section').css('padding-top');
								var sBot = jQuery(this).closest('.elementor-section').css('padding-bottom');
								var sHeight = jQuery(this).closest('.elementor-section').innerHeight();
								var svert = sHeight - (parseFloat(sTop) + parseFloat(sBot));

								if (typeof svert !== 'undefined' && svert < 5) {
									jQuery(this).closest('.elementor-section').hide();
								}
							});
						}

					});
				}
			})(jQuery, window, document);
		</script>
<?php }
}
?>