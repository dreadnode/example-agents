<?php

/**
 * Add Widget Options in Gutenberg
 *
 * Process Managing of Widget Options.
 *
 * @copyright   Copyright (c) 2023, Boholweb WP
 * @since       5.0.1
 */

use Automattic\Jetpack\Sync\Functions;

if (!defined('ABSPATH')) exit; // Exit if accessed directly

function widgetopts_toolbar_scripts()
{
	global $widget_options, $pagenow;

	if (($pagenow != 'widgets.php' && $pagenow != 'customize.php') &&
		(!isset($widget_options["hide_page_and_post_block"]) || $widget_options["hide_page_and_post_block"] != "activate") ||
		($widget_options["hide_page_and_post_block"] == "activate" && isset($widget_options["settings"]) &&
			isset($widget_options["settings"]["hide_page_and_post_block"]) &&
			$widget_options["settings"]["hide_page_and_post_block"]["page_and_post_block"] == "1")
	) {
		//do nothing
	} else {
		wp_register_script(
			'widgetopts-gutenberg-toolbar',
			WIDGETOPTS_PLUGIN_URL . 'includes/widgets/gutenberg/build/index.js',
			['wp-blocks', 'wp-components', 'wp-element', 'wp-i18n', 'wp-block-editor'],
			filemtime(WIDGETOPTS_PLUGIN_DIR . 'includes/widgets/gutenberg/build/index.js')
		);
		wp_enqueue_script('widgetopts-gutenberg-toolbar');
		wp_localize_script('widgetopts-gutenberg-toolbar', 'widgetoptsGutenberg', array(
			'ajaxurl' => admin_url('admin-ajax.php'),
		));
	}
}
add_action('enqueue_block_editor_assets', 'widgetopts_toolbar_scripts');

function widgetopts_check_widget_editor_referer($customizer = false)
{
	// Check if the HTTP referer is set
	if (isset($_SERVER['HTTP_REFERER'])) {
		// Parse the referer URL
		$referer_url = parse_url($_SERVER['HTTP_REFERER']);

		$path = $customizer === true ? '/wp-admin/customize.php' : '/wp-admin/widgets.php';
		// Check if the referer URL is from the widget editor
		if (strpos($referer_url['path'], $path) !== false) {
			// The request likely came from the widget editor
			return true;
		}
	}

	return false;
}

add_action('rest_api_init', function () {
	global $wp_widget_factory;

	if (isset($wp_widget_factory) && isset($wp_widget_factory->widgets)) {
		// Modify the show_instance_in_rest option for each registered widget type
		foreach ($wp_widget_factory->widgets as $widget) {
			if (isset($widget->widget_options) && is_array($widget->widget_options)) {
				// if (stristr($widget->id, 'tribe-widget-events-list')) {
				$widget->widget_options['show_instance_in_rest'] = true;
				// }
			}
		}
	}
}, 999);

add_filter('register_block_type_args', function ($args, $name) {
	global $orig_callback;
	$plugin_names = apply_filters('widget_options_type_string_attribute', ['luckywp/tableofcontents']);
	if (in_array($name, $plugin_names)) {
		$args['attributes']['extended_widget_opts_block'] = array(
			'type' => 'string',
			'default' => json_encode((object)[])
		);

		$args['attributes']['extended_widget_opts'] = array(
			'type' => 'string',
			'default' => json_encode((object)[])
		);
	} else if ($name == 'events-calendar-shortcode/block') {
		$args['attributes']['extended_widget_opts_block'] = array(
			'type' => 'object'
		);

		$args['attributes']['extended_widget_opts'] = array(
			'type' => 'object'
		);

		if (isset($args['render_callback'])) {
			$orig_callback = $args['render_callback'];
			$args['render_callback'] = function ($attributes) {
				global $orig_callback;
				if (isset($attributes['extended_widget_opts'])) {
					$attributes['extended_widget_opts'] = json_encode($attributes['extended_widget_opts']);
				}

				if (isset($attributes['extended_widget_opts_block'])) {
					$attributes['extended_widget_opts_block'] = json_encode($attributes['extended_widget_opts_block']);
				}

				if (function_exists('ecs_render_block')) {
					return ecs_render_block($attributes);
				}
				return call_user_func($orig_callback, $attributes);
			};
		}
	} else if (stripos($name, 'jetpack') !== false) {
		$args['attributes']['extended_widget_opts_block'] = array(
			'type' => 'object',
			// 'default' => (object)[]
		);

		$args['attributes']['extended_widget_opts'] = array(
			'type' => 'object',
			// 'default' => (object)[]
		);
	} else {
		//if block type is not luckywp/tableofcontents use type object
		$args['attributes']['extended_widget_opts_block'] = array(
			'type' => 'object',
			'default' => (object)[]
		);

		$args['attributes']['extended_widget_opts'] = array(
			'type' => 'object',
			'default' => (object)[]
		);
	}

	$args['attributes']['extended_widget_opts_state'] = array(
		'type' => 'string',
		'default' => ''
	);

	$args['attributes']['extended_widget_opts_clientid'] = array(
		'type' => 'string',
		'default' => ''
	);

	$args['attributes']['dateUpdated'] = array(
		'type' => 'string',
		'default' => ''
	);

	return $args;
}, 50, 2);

add_filter('widget_types_to_hide_from_legacy_widget_block', function ($notAllowed) {
	return array();
}, 99999, 1);

if (wp_use_widgets_block_editor()) {
	add_filter('widget_update_callback', function ($instance, $new_instance, $old_instance, $obj) {

		$base = explode('-', $obj->id);
		//this is for block widget saving
		if (stristr($base[0], 'block') || $base[0] == 'block') {
			//remove widgetopts attribute from blocks when it is classic editor
			if (!empty($new_instance['content'])) {
				$block = parse_blocks($new_instance['content']);
				if (!empty($block[0]) && !empty($block[0]['attrs'])) {
					if (!empty($block[0]['attrs']['extended_widget_opts_block'])) {
						$instance['extended_widget_opts-' . $obj->id] = $block[0]['attrs']['extended_widget_opts_block'];

						unset($block[0]['attrs']['extended_widget_opts_block']);
						$instance['content'] = serialize_blocks($block);
					}
				}
			}
		} else {
			//this is for legacy widget saving
			//for first time saving of legacy widget, after this frontend will hanlde the saving in block editor
			//for already exist widget
			if (!empty($new_instance['extended_widget_opts-' . $obj->id])) {
				$instance['extended_widget_opts-' . $obj->id] = $new_instance['extended_widget_opts-' . $obj->id];
			}

			//for new created widget
			if (isset($new_instance['extended_widget_opts-undefined']) && empty($new_instance['extended_widget_opts-' . $obj->id])) {
				$instance['extended_widget_opts-' . $obj->id] = $new_instance['extended_widget_opts-undefined'];

				array_pop($base);
				$new_base = implode('-', $base);

				$instance['extended_widget_opts-' . $obj->id]['id_base'] = $base === false ? '-1' : $new_base;
				unset($new_instance['extended_widget_opts-undefined']);
				if (isset($instance['extended_widget_opts-undefined'])) {
					unset($instance['extended_widget_opts-undefined']);
				}
			}
		}

		return $instance;
	}, 100, 4);
}

add_filter('render_block', function ($block_content, $parsed_block, $obj) {
	if (!is_admin()) {
		add_filter("render_block_{$obj->name}", "blockopts_filter_before_display", 100, 3);
	}
	return $block_content;
}, 100, 3);

function blockopts_filter_before_display($block_content, $parsed_block, $obj)
{
	//for freeform filter and assignment
	if (is_null($parsed_block['blockName']) || empty($parsed_block['blockName'])) {
		if (!isset($parsed_block['attrs']) || empty($parsed_block['attrs'])) {
			$result = null;
			if (isset($parsed_block['innerContent']) && !empty($parsed_block['innerContent'][0])) {
				$is_okay = preg_match("/<!--start_widgetopts[\s]+[{\":,}\w\W]*[\s]+end_widgetopts-->/", $parsed_block['innerContent'][0], $result);
				if ($is_okay === 1) {
					if (!is_null($result) && is_array($result)) {
						$content = str_replace('<!--start_widgetopts', '', $result[0]);
						$content = str_replace('end_widgetopts-->', '', $content);
						$content = trim($content);
						$parsed_block['attrs']['extended_widget_opts'] = json_decode($content, true);
					}
				}
			}
		}
	}

	if (isset($parsed_block) && isset($parsed_block['attrs']) && (isset($parsed_block['attrs']['extended_widget_opts']) || isset($parsed_block['attrs']['extended_widget_opts_block']))) {
		global $widget_options, $current_user;
		$instance = $parsed_block['attrs'];

		//if idbase is not -1 it is a widget
		if (isset($parsed_block['attrs']['extended_widget_opts']) && isset($parsed_block['attrs']['extended_widget_opts']['id_base']) && $parsed_block['attrs']['extended_widget_opts']['id_base'] != -1) {
			return $block_content;
		}

		// WPML FIX
		$hasWPML = has_filter('wpml_current_language');
		$hasWPML = (function_exists('pll_the_languages')) ? false : $hasWPML;
		$default_language = $hasWPML ? apply_filters('wpml_default_language', NULL) : false;

		$hidden     = false;
		$opts       = (isset($instance['extended_widget_opts'])) ? $instance['extended_widget_opts'] : (isset($instance['extended_widget_opts_block']) ? $instance['extended_widget_opts_block'] : array());
		$visibility = array('show' => array(), 'hide' => array());
		$tax_opts   = (isset($widget_options['settings']) && isset($widget_options['settings']['taxonomies_keys'])) ? $widget_options['settings']['taxonomies_keys'] : array();

		$visibility         = isset($opts['visibility']) ? $opts['visibility'] : array();
		$visibility_opts    = isset($opts['visibility']['options']) ? $opts['visibility']['options'] : 'hide';
		$authorPageSelection = "";

		//wordpress pages
		$is_misc    = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['misc'])) ? true : false;
		$is_types   = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['post_type'])) ? true : false;
		$is_tax     = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['taxonomies'])) ? true : false;
		$is_inherit = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['inherit'])) ? true : false;

		//WOOCOMMERCE
		$isWooPage = false;
		if (class_exists('WooCommerce')) {
			$wooPageID = 0;

			$wooPageID = (is_shop()) ? get_option('woocommerce_shop_page_id') : $wooPageID;
			if ($wooPageID) {
				$isWooPage = true;

				$visibility['pages'] = !empty($visibility['pages']) ? $visibility['pages'] : [];
				if ($visibility_opts == 'hide' && (array_key_exists($wooPageID, $visibility['pages']) || in_array($wooPageID, $visibility['pages']))) {
					$hidden = true; //hide if exists on hidden pages
				} elseif ($visibility_opts == 'show' &&  (!array_key_exists($wooPageID, $visibility['pages']) && !in_array($wooPageID, $visibility['pages']))) {
					$hidden = true; //hide if doesn't exists on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_page_block', $hidden);

				if ($hidden) {
					return false;
				}
			}
		}

		// Normal Pages
		if (!$isWooPage) {
			if ($is_misc && ((is_home() && is_front_page()) || is_front_page())) {
				if (isset($visibility['misc']['home']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif (!isset($visibility['misc']['home']) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				if (isset($visibility['misc']['home']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_home_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_home()) { //filter for blog page
				if (isset($visibility['misc']['blog']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif (!isset($visibility['misc']['blog']) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				if (isset($visibility['misc']['blog']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_blog_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_tax && is_category() && is_array($tax_opts) && in_array('category', $tax_opts)) {
				if (!isset($visibility['categories'])) {
					$visibility['categories'] = array();
				}

				$cat_lists = array();

				if (isset($visibility['tax_terms']['category'])) {
					$cat_lists = $visibility['tax_terms']['category'];
				} elseif (isset($visibility['categories'])) {
					$cat_lists = $visibility['categories'];
				}

				// WPML TRANSLATION OBJECT FIX
				$category_id = ($hasWPML) ? apply_filters('wpml_object_id', get_query_var('cat'), 'category', true, $default_language) : get_query_var('cat');

				if (!isset($visibility['taxonomies']['category']) && $visibility_opts == 'hide' && (array_key_exists($category_id, $cat_lists) || in_array($category_id, $cat_lists))) {
					$hidden = true; //hide if exists on hidden pages
				} elseif (!isset($visibility['taxonomies']['category']) && $visibility_opts == 'show' && (!array_key_exists($category_id, $cat_lists) && !in_array($category_id, $cat_lists))) {
					$hidden = true; //hide if doesn't exists on visible pages
				} elseif (isset($visibility['taxonomies']['category']) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all categories
				} elseif (isset($visibility['taxonomies']['category']) && $visibility_opts == 'show') {
					$hidden = false; //hide to all categories
				}

				if (isset($visibility['taxonomies']['category']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_categories_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_tax && is_tag() && is_array($tax_opts) && in_array('post_tag', $tax_opts)) {
				if (!isset($visibility['tags'])) {
					$visibility['tags'] = array();
				}

				$tag_lists = (isset($visibility['tax_terms']['post_tag'])) ? $visibility['tax_terms']['post_tag'] : array();

				// WPML TRANSLATION OBJECT FIX
				$tag_id = ($hasWPML) ? apply_filters('wpml_object_id', get_query_var('tag_id'), 'post_tag', true, $default_language) : get_query_var('tag_id');

				if (!isset($visibility['taxonomies']['post_tag']) && $visibility_opts == 'hide' && (array_key_exists($tag_id, $tag_lists) || in_array($tag_id, $tag_lists))) {
					$hidden = true; //hide if exists on hidden pages
				} elseif (!isset($visibility['taxonomies']['post_tag']) && $visibility_opts == 'show' && (!array_key_exists($tag_id, $tag_lists) && !in_array($tag_id, $tag_lists))) {
					$hidden = true; //hide if doesn't exists on visible pages
				} elseif (isset($visibility['taxonomies']['post_tag']) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all tags
				} elseif (isset($visibility['taxonomies']['post_tag']) && $visibility_opts == 'show') {
					$hidden = false; //hide to all tags
				}

				if (isset($visibility['taxonomies']['post_tag']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_tags_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_tax && is_tax()) {
				$term = get_queried_object();
				$term_lists = array();

				if (isset($visibility['tax_terms']) && isset($visibility['tax_terms'][$term->taxonomy])) {
					$term_lists = $visibility['tax_terms'][$term->taxonomy];
				}

				// WPML TRANSLATION OBJECT FIX
				$term_id = ($hasWPML) ? apply_filters('wpml_object_id', $term->term_id, $term->taxonomy, true, $default_language) : $term->term_id;

				if (isset($visibility['taxonomies']) && !isset($visibility['taxonomies'][$term->taxonomy]) && $visibility_opts == 'hide' && (array_key_exists($term_id, $term_lists) || in_array($term_id, $term_lists))) {
					$hidden = true; //hide if exists on hidden pages
				} elseif (isset($visibility['taxonomies']) && !isset($visibility['taxonomies'][$term->taxonomy]) && $visibility_opts == 'show' && (!array_key_exists($term_id, $term_lists) && !in_array($term_id, $term_lists))) {
					$hidden = true; //hide if doesn't exists on visible pages
				} elseif (isset($visibility['taxonomies']) &&  isset($visibility['taxonomies'][$term->taxonomy]) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all tags
				} elseif (isset($visibility['taxonomies']) && isset($visibility['taxonomies'][$term->taxonomy]) && $visibility_opts == 'show') {
					$hidden = false; //hide to all tags
				} elseif (isset($visibility['tax_terms']) && isset($visibility['tax_terms'][$term->taxonomy]) && $visibility_opts == 'hide' && (array_key_exists($term_id, $term_lists) || in_array($term_id, $term_lists))) {
					$hidden = true; //hide if exists on hidden pages
				} elseif (isset($visibility['tax_terms']) && isset($visibility['tax_terms'][$term->taxonomy]) && $visibility_opts == 'show' && (array_key_exists($term_id, $term_lists) || in_array($term_id, $term_lists))) {
					$hidden = false; //hide if doesn't exists on visible pages
				} elseif (!isset($visibility['taxonomies']) && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on hidden pages
				}

				if (isset($visibility['taxonomies']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_taxonomies_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_archive()) {
				if (isset($visibility['misc']['archives']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif (!isset($visibility['misc']['archives']) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				} else if (isset($visibility['misc']['archives']) && (!empty($authorPageSelection) && $authorPageSelection == '3') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_archives_block', $hidden);

				if ($hidden) {
					return false;
				}
			} elseif (is_post_type_archive()) {
				if (!isset($visibility['types']) || ($is_types && !isset($visibility['types']))) {
					$visibility['types'] = array();
				}

				$current_type_archive = get_post_type();
				if (!empty($current_type_archive)) {
					if ($visibility_opts == 'hide' && array_key_exists($current_type_archive, $visibility['types'])) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !array_key_exists($current_type_archive, $visibility['types'])) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				}

				if (array_key_exists($current_type_archive, $visibility['types']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_post_type_archive_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_404()) {
				if (isset($visibility['misc']['404']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif (!isset($visibility['misc']['404']) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				if (isset($visibility['misc']['404']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_404_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_search()) {
				if (isset($visibility['misc']['search']) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif (!isset($visibility['misc']['search']) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				if (isset($visibility['misc']['search']) && (!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_search_block', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif (is_single() && !is_page()) {
				global $post;
				if (!isset($visibility['types']) || ($is_types && !isset($visibility['types']))) {
					$visibility['types'] = array();
				}
				if ($visibility_opts == 'hide' && array_key_exists($post->post_type, $visibility['types'])) {
					$hidden = true; //hide if exists on hidden pages
				} elseif ($visibility_opts == 'show' && !array_key_exists($post->post_type, $visibility['types'])) {
					$hidden = true; //hide if doesn't exists on visible pages
				}

				if ((!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				// do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_types_block', $hidden);

				$taxonomy_names  = get_post_taxonomies();
				$array_intersect = array_intersect($tax_opts, $taxonomy_names);
				// print_r( $tax_opts );
				if (!isset($visibility['tax_terms']['category']) && isset($visibility['categories'])) {
					$visibility['tax_terms']['category'] = $visibility['categories'];
				}

				// WPML FIX
				$postID = ($hasWPML) ? apply_filters('wpml_object_id', $post->ID, $post->post_type, true, $default_language) : $post->ID;

				if (!empty($array_intersect)) {
					foreach ($array_intersect  as $tax_key => $tax_value) {
						if (in_array($tax_value, $tax_opts) && isset($visibility['tax_terms']) && isset($visibility['tax_terms'][$tax_value]) && !empty($visibility['tax_terms'][$tax_value])) {
							$term_list = wp_get_post_terms($postID, $tax_value, array("fields" => "ids"));

							// WPML TRANSLATION OBJECT FIX
							if ($hasWPML) {
								$temp_term_list = [];
								foreach ($term_list as $index => $termID) {
									$temp_term_list[] = apply_filters('wpml_object_id', $termID, $tax_value, true, $default_language);
								}
								$term_list = (!empty($temp_term_list)) ? $temp_term_list : $term_list;
							}

							if (is_array($term_list) && !empty($term_list)) {
								$checked_terms   = array_keys($visibility['tax_terms'][$tax_value]);
								$checked_terms = (intval($checked_terms[0]) == 0) ? $visibility['tax_terms'][$tax_value] : $checked_terms;
								$intersect      = array_intersect($term_list, $checked_terms);
								if (!empty($intersect) && $visibility_opts == 'hide') {
									$hidden = true;
								} elseif (!empty($intersect) && $visibility_opts == 'show') {
									$hidden = false;
								}
							}
						}
						// do return to bypass other conditions
						$hidden = apply_filters('widget_options_visibility_single_block_' . $tax_value, $hidden);
					}
				}


				if ($hidden) {
					return false;
				}
				// echo $type;
			} elseif ($is_types && is_page()) {
				global $post;

				//do post type condition first
				if (isset($visibility['types']) && isset($visibility['types']['page'])) {
					if ($visibility_opts == 'hide' && array_key_exists('page', $visibility['types'])) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !array_key_exists('page', $visibility['types'])) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				} else {
					//do per pages condition
					if (!isset($visibility['pages'])) {
						$visibility['pages'] = array();
					}

					// WPML FIX
					$page_id = get_queried_object_id();
					$parent_id = wp_get_post_parent_id($page_id);

					$pageID = ($hasWPML) ? apply_filters('wpml_object_id', $page_id, 'page', true, $default_language) : $page_id;
					$parentID = ($hasWPML) ? apply_filters('wpml_object_id', $parent_id, 'page', true, $default_language) : $parent_id;

					$page_in_array = in_array($pageID, $visibility['pages']);
					//for the compatibility of the data of lower version 3.8.10 and below
					if (array_key_exists($pageID, $visibility['pages'])) {
						if ($visibility['pages'][$pageID] == 1) {
							$page_in_array = true;
						}
					}

					//add parent inherit option
					if ($is_inherit && $parentID && (array_key_exists($parentID, $visibility['pages']) || in_array($pageID, $visibility['pages']))) {
						$visibility['pages'][] = $pageID;
						// print_r( $visibility['pages'] );
					}

					if ($visibility_opts == 'hide' && $page_in_array) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !$page_in_array) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				}

				if ((!empty($authorPageSelection) && $authorPageSelection == '2') && $visibility_opts == 'show') {
					$hidden = true; //hide if checked on visible pages but the visibilty_opts is show
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widget_options_visibility_page_block', $hidden);
				if ($hidden) {
					return false;
				}
			}
		}
		//end wordpress pages

		//ACF
		if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) {
			if (isset($visibility['acf']['field']) && !empty($visibility['acf']['field'])) {
				$acf = get_field_object($visibility['acf']['field']);
				if ($acf && is_array($acf)) {
					$acf_visibility    = (isset($visibility['acf']) && isset($visibility['acf']['visibility'])) ? $visibility['acf']['visibility'] : 'hide';
					//handle repeater fields
					if (isset($acf['value'])) {
						if (is_array($acf['value'])) {
							$acf['value'] = implode(', ', array_map(function ($acf_array_value) {
								if (!is_array($acf_array_value)) return $acf_array_value;

								$acf_implode = implode(',', array_filter($acf_array_value));
								return $acf_implode;
							}, $acf['value']));
						}
					}
					switch ($visibility['acf']['condition']) {
						case 'equal':
							if (isset($acf['value'])) {
								if ('show' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
									$hidden = false;
								} else if ('show' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
									$hidden = true;
								} else if ('hide' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
									$hidden = true;
								} else if ('hide' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
									$hidden = false;
								}
							}
							break;
						case 'not_equal':
							if (isset($acf['value'])) {
								if ('show' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
									$hidden = true;
								} else if ('show' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
									$hidden = false;
								} else if ('hide' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
									$hidden = false;
								} else if ('hide' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
									$hidden = true;
								}
							}
							break;
						case 'contains':
							if (isset($acf['value'])) {
								if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
									$hidden = false;
								} else if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
									$hidden = true;
								} else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
									$hidden = true;
								} else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
									$hidden = false;
								}
							}
							break;
						case 'not_contains':
							if (isset($acf['value'])) {
								if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
									$hidden = true;
								} else if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
									$hidden = false;
								} else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
									$hidden = false;
								} else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
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
					$hidden = apply_filters('widget_options_visibility_acf_block', $hidden);
					if ($hidden) {
						return false;
					}
				}
			}
		}

		//login state
		if (isset($widget_options['state']) && 'activate' == $widget_options['state'] && isset($opts['roles'])) {
			if (isset($opts['roles']['state']) && !empty($opts['roles']['state'])) {
				//do state action here
				if ($opts['roles']['state'] == 'out' && is_user_logged_in()) {
					return false;
				} else if ($opts['roles']['state'] == 'in' && !is_user_logged_in()) {
					return false;
				}
			}
		}

		if ('activate' == $widget_options['logic']) {
			// display widget logic
			if (isset($opts['class']) && isset($opts['class']['logic']) && !empty($opts['class']['logic'])) {
				$display_logic = stripslashes(trim($opts['class']['logic']));
				$display_logic = apply_filters('widget_options_logic_override_block', $display_logic);
				$display_logic = apply_filters('extended_widget_options_logic_override_block', $display_logic);
				if ($display_logic === false) {
					return false;
				}
				if ($display_logic === true) {
					return true;
				}
				// if (stristr($display_logic, "return") === false) {
				// 	$display_logic = "return (" . $display_logic . ");";
				// }
				$display_logic = htmlspecialchars_decode($display_logic, ENT_QUOTES);
				if (!widgetopts_safe_eval($display_logic)) {
					return false;
				}
			}
		}

		if ('activate' == $widget_options['hide_title']) {
			//hide widget title
			if (isset($instance['title']) && isset($opts['class']) && isset($opts['class']['title']) && '1' == $opts['class']['title']) {
				$instance['title'] = '';
			}
		}

		$block_content = widgetopts_add_classes_post_block($block_content, $parsed_block, $obj);
	}

	return $block_content;
}

/*
 * Add custom classes on widget
 */
function widgetopts_add_classes_post_block($block_content, $parsed_block, $obj)
{
	global $widget_options, $wp_registered_widget_controls;
	$classe_to_add  = '';
	$id_to_add = '';
	$widget_id_set = '';
	$data_attr = '';
	$instance       = $parsed_block['attrs'];

	if (isset($instance)) {
		$opts           = (isset($instance['extended_widget_opts'])) ? $instance['extended_widget_opts'] : (isset($instance['extended_widget_opts_block']) ? $instance['extended_widget_opts_block'] : array());
	} else {
		$opts = array();
	}

	$custom_class   = isset($opts['class']) ? $opts['class'] : '';

	if ('activate' == $widget_options['classes'] && isset($widget_options['settings']['classes'])) {
		//don't add the IDs when the setting is set to NO
		if (isset($widget_options['settings']['classes']['id'])) {
			if (is_array($custom_class) && isset($custom_class['id']) && !empty($custom_class['id'])) {
				$id_to_add = $custom_class['id'];
				$widget_id_set = $custom_class['id'];
			}
		}
	}

	$get_classes = widgetopts_classes_generator($opts, $widget_options, $widget_options['settings']);

	//double check array
	if (!is_array($get_classes)) {
		$get_classes = array();
	}

	if (!empty($get_classes)) {
		$classe_to_add .= (implode(' ', $get_classes)) . ' ';
		//$block_content = preg_replace('class="', $classes, $block_content, 1);
	}

	// $params[0]['before_widget'] = str_replace('class="', ' data-animation="asdf" class="', $params[0]['before_widget']);

	$match = [];
	$has_match = preg_match('/<\w*[^>]*>/', $block_content, $match);

	if ($has_match == 1) {
		if (!empty($id_to_add)) {
			$has_match_id = preg_match('/[id="]/', $match[0]);
			if ($has_match_id == 1) {
				$block_content = preg_replace('/id="[^"]*/', "id=\"{$id_to_add}", $block_content, 1);
			} else {
				$block_content = preg_replace('/>/', " id=\"{$id_to_add}\">", $block_content, 1);
			}
		}

		if (!empty($classe_to_add)) {
			$has_match_class = preg_match('/[class="]/', $match[0]);
			if ($has_match_class == 1) {
				$block_content = preg_replace('/class="/', "class=\"{$classe_to_add}", $block_content, 1);
			} else {
				$block_content = preg_replace('/>/', " class=\"{$classe_to_add}\">", $block_content, 1);
			}
		}

		if (!empty($data_attr)) {
			$block_content = preg_replace('/>/', " {$data_attr}>", $block_content, 1);
		}
	}

	return $block_content;
}


/**
 * Gutenberg ajax functions
 */
function widgetopts_get_types()
{
	global $widgetopts_types;
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	wp_send_json_success(((!empty($widgetopts_types)) ? $widgetopts_types : widgetopts_global_types()));
	die;
}
add_action('wp_ajax_widgetopts_get_types', 'widgetopts_get_types');


function widgetopts_get_taxonomies()
{
	global $widgetopts_taxonomies;
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	wp_send_json_success(((!empty($widgetopts_taxonomies)) ? $widgetopts_taxonomies : widgetopts_global_taxonomies()));
	die;
}
add_action('wp_ajax_widgetopts_get_taxonomies', 'widgetopts_get_taxonomies');

function widgetopts_acf_get_field_groups()
{
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	$fields = array();
	if (function_exists('acf_get_field_groups')) {
		$groups = acf_get_field_groups();
		if (is_array($groups)) {
			foreach ($groups as $group) {
				$fields[$group['ID']] = array('title' => $group['title'], 'fields' => acf_get_fields($group));
			}
		}
	} else {
		$groups = apply_filters('acf/get_field_groups', array());
		if (is_array($groups)) {
			foreach ($groups as $group) {
				$fields[$group['id']] = array('title' => $group['title'], 'fields' => apply_filters('acf/field_group/get_fields', array(), $group['id']));
			}
		}
	}

	wp_send_json_success($fields);
	die;
}
add_action('wp_ajax_widgetopts_acf_get_field_groups', 'widgetopts_acf_get_field_groups');

function widgetopts_get_legacy_data()
{
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	if (isset($_POST['id_base'])) {
		wp_send_json_success(array());
		die;
	}
	$settings = array();
	$settings = get_option('widget_' + sanitize_key($_POST['id_base']));

	if (false === $settings) {
		return $settings;
	}

	if (!is_array($settings) && !($settings instanceof ArrayObject || $settings instanceof ArrayIterator)) {
		$settings = array();
	}

	wp_send_json_success($settings);
	die;
}
add_action('wp_ajax_widgetopts_get_legacy_data', 'widgetopts_get_legacy_data');

function widgetopts_get_settings_ajax()
{
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}
	$settings = widgetopts_get_settings();

	if (!current_user_can('administrator')) {
		$settings['logic'] = 'deactivate';
	}

	wp_send_json_success($settings);
	die;
}
add_action('wp_ajax_widgetopts_get_settings_ajax', 'widgetopts_get_settings_ajax');

function widgetopts_get_pages()
{
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	$pages = [];

	$pargs = array(
		'hierarchical' => true,
		'child_of' => 0, // Display all pages regardless of parent
		'parent' => -1, // Display all pages regardless of parent
		'sort_order' => 'ASC',
		'sort_column' => 'menu_order, post_title'
	);

	$pageLoop = get_pages($pargs);

	if ($pageLoop) {
		foreach ($pageLoop as $objPage) {
			$depth = count(get_ancestors($objPage->ID, 'page'));
			// Determine indentation for hierarchical display
			$indent = str_repeat('-', $depth);
			$objPage->post_title = $indent . "" . $objPage->post_title;
		}
		$pages = $pageLoop;
	}

	wp_send_json_success($pages);
	die;
}
add_action('wp_ajax_widgetopts_get_pages', 'widgetopts_get_pages');

function widgetopts_get_terms()
{
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	$terms = array();

	$_terms = array();
	$_terms = get_terms();

	if (!is_wp_error($_terms)) {
		foreach ($_terms as $t) {
			$terms[$t->name][] = $t;
		}
	}

	wp_send_json_success($terms);
	die;
}
add_action('wp_ajax_widgetopts_get_terms', 'widgetopts_get_terms');

function widgetopts_get_users()
{
	global $wp_version;

	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}

	$authors = array();

	$args = array();

	$is_6_3_and_above = version_compare($wp_version, '6.3', '>=');
	if ($is_6_3_and_above) {
		$args['cache_results'] = apply_filters('cache_widgetopts_ajax_taxonomy_search', true);
	}

	$_authors  = get_users($args);

	if (!empty($_authors)) {

		if (is_iterable($_authors)) {
			foreach ($_authors as $a) {
				$displayname = isset($a->display_name) ? $a->display_name : (isset($a->data) && isset($a->data->display_name) ? $a->data->display_name : '');
				$authors[] = ["ID" => $a->ID, "display_name" => $displayname];
			}
		}
	}

	wp_send_json_success($authors);
	die;
}
add_action('wp_ajax_widgetopts_get_users', 'widgetopts_get_users');

function widgetopts_ajax_roles_search_block()
{
	if (!(current_user_can('edit_pages') || current_user_can('edit_posts') || current_user_can('edit_theme_options'))) {
		die;
	}
	$response = [
		'results' => [],
		'pagination' => ['more' => false]
	];

	$term = isset($_POST['term']) && !empty($_POST['term']) ? $_POST['term'] : '';

	$roles = get_editable_roles();
	if (!empty($roles)) {
		foreach ($roles as $role_name => $role_info) {
			// if ((!empty($term) && stristr($role_name, $term) !== false) || stristr($role_name, $role_info['name']) !== false) {
			$response['results'][] = [
				'id' => $role_name,
				'text' => $role_info['name']
			];
			// }
		}
	}

	wp_send_json_success($response);
	die;
}
add_action('wp_ajax_widgetopts_ajax_roles_search_block',  'widgetopts_ajax_roles_search_block');
