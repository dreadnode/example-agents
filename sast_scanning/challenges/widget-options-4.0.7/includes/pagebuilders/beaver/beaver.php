<?php

/**
 * Plugin Name: Widget Options for Beaver Builder
 * Plugin URI: https://widget-options.com/
 * Description: <strong>Requires <a href="https://wordpress.org/plugins/widget-options/" target="_blank">Widget Options Plugin</a></strong>! Extend functionalities to Beaver Builder for more visibility restriction options.
 * Version: 1.0
 * Author: Widget Options Team
 * Author URI: https://widget-options.com/
 * Text Domain: widget-options
 * Domain Path: languages
 *
 * @category Widgets
 * @author Widget Options Team
 * @version 1.0
 */
// Exit if accessed directly.
if (!defined('ABSPATH')) exit;
if (!class_exists('WP_Widget_Options_Beaver')) :

	/**
	 * Main WP_Widget_Options_Beaver Class.
	 *
	 * @since 1.0
	 */
	class WP_Widget_Options_Beaver
	{

		public static function init()
		{
			$class = __CLASS__;
			new $class;
		}

		function __construct()
		{
			global $widget_options;

			add_filter('fl_builder_register_settings_form', array(&$this, 'widgetopts_beaver_settings'), 10, 2);
			add_action('fl_builder_control_widgetopts-beaver-tabnav', array(&$this, 'fl_widgetopts_beaver_tabnav'), 1, 4);
			add_action('wp_enqueue_scripts', array(&$this, 'fl_widgetopts_beaver_scripts'));
			add_action('fl_builder_control_widgetopts-select2', array(&$this, 'fl_widgetopts_beaver_select2'), 1, 4);
			add_action('fl_builder_control_widgetopts-upgrade', array(&$this, 'fl_widgetopts_upgrade'), 1, 4);
			// add_action( 'admin_notices', array( &$this, 'widgetopts_plugin_check' ) );

			add_filter('fl_builder_is_node_visible', array(&$this, 'widgetopts_beaver_is_node_visible'), 10, 2);
		}

		function widgetopts_beaver_settings($form, $id)
		{
			if (!isset($form['widgetopts']) && !is_admin()) {
				//fix not registering global values
				if (!function_exists('widgetopts_register_globals')) {
					require_once WIDGETOPTS_PLUGIN_DIR . 'includes/admin/globals.php';
					widgetopts_register_globals();
				}

				global $widget_options, $widgetopts_taxonomies, $widgetopts_types, $widgetopts_categories;
				$widgetopts_pages 		= widgetopts_global_pages();


				$sections 	= array();
				$pages      = (!empty($widgetopts_pages))       ? $widgetopts_pages         : array();
				$taxonomies = (!empty($widgetopts_taxonomies))  ? $widgetopts_taxonomies    : array();
				$types      = (!empty($widgetopts_types))       ? $widgetopts_types         : array();
				$categories = (!empty($widgetopts_categories))  ? $widgetopts_categories    : array();

				$get_terms = array();
				if (!empty($widget_options['settings']['taxonomies']) && is_array($widget_options['settings']['taxonomies'])) {
					foreach ($widget_options['settings']['taxonomies'] as $tax_opt => $vall) {
						$tax_name = 'widgetopts_taxonomy_' . str_replace('-', '__', $tax_opt);
						// global $$tax_name;
						if (isset($GLOBALS[$tax_name])) {
							$get_terms[$tax_opt] = $GLOBALS[$tax_name];
						}
					}
				}

				// print_r( $pages ); die();

				$sections['widgetopts-fields'] = array(
					'fields' 	  => array(
						'widgetopts-tabnav' => array(
							'type'          => 'widgetopts-beaver-tabnav',
						)
					)
				);

				if (isset($widget_options['visibility']) && 'activate' == $widget_options['visibility']) {
					$visibility_fld = array();

					$visibility_fld['widgetopts_visibility_show'] = array(
						'type'          => 'select',
						'label' 		=> __('Show or Hide', 'widget-options'),
						'options'       => array(
							'hide'      => __('Hide on Selected Pages', 'widget-options'),
							'show'      => __('Show on Selected Pages', 'widget-options')
						)
					);

					if (isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['post_type']) && '1' == $widget_options['settings']['visibility']['post_type']) {

						if (!empty($pages)) {
							$pages_array 	= array();
							foreach ($pages as $page) {
								$pages_array[$page->ID] = $page->post_title;
							}

							$visibility_fld['widgetopts_visibility_pages'] = array(
								'type'				=> 'widgetopts-select2',
								'label'				=> __('Pages', 'widget-options'),
								'options'			=> $pages_array,
								'multi-select' 		=> true,
								'description' 		=> __('Click to search or select', 'widget-options'),
							);
						}

						if (!empty($types)) {
							$types_array = array();
							foreach ($types as $ptype => $type) {
								$types_array[$ptype] = $type->labels->name;
							}

							$visibility_fld['widgetopts_visibility_types'] = array(
								'type'				=> 'widgetopts-select2',
								'label'				=> __('Post Types', 'widget-options'),
								'options'			=> $types_array,
								'multi-select' 		=> true,
								'description' 		=> __('Click to search or select', 'widget-options')
							);
						}
					}

					if (isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['taxonomies']) && '1' == $widget_options['settings']['visibility']['taxonomies']) {
						//          if( !empty( $widget_options['settings']['taxonomies'] ) && is_array( $widget_options['settings']['taxonomies'] ) ){
						//              foreach ( $widget_options['settings']['taxonomies'] as $tax_opt => $vallue ) {
						//                  $term_array = array();
						//                  // if( !empty( $get_terms ) ){
						//                  	foreach ( $get_terms[ $tax_opt ] as $get_term ) {
						//                       $term_array[ $get_term->term_id ] = $get_term->name;
						//                   }

						//                   $visibility_fld['widgetopts_visibility_tax_'. $tax_opt] = array(
						// 	'type'				=> 'widgetopts-select2',
						// 	'label'				=> $taxonomies[ $tax_opt ]->label,
						// 	'options'			=> $term_array,
						// 	'multi-select' 		=> true,
						// 	'description' 		=> __( 'Click to search or select', 'widget-options' )
						// );
						//                  // }
						//              }
						//          }

						if (!empty($categories)) {
							$cat_array = array();
							foreach ($categories as $cat) {
								$cat_array[$cat->cat_ID] = $cat->cat_name;
							}

							$visibility_fld['widgetopts_visibility_tax_category'] = array(
								'type'				=> 'widgetopts-select2',
								'label'				=> __('Categories', 'widget-options'),
								'options'			=> $cat_array,
								'multi-select' 		=> true,
								'description' 		=> __('Click to search or select', 'widget-options')
							);
						}

						if (!empty($taxonomies)) {
							$tax_array = array();
							foreach ($taxonomies as $taxonomy) {
								$tax_array[$taxonomy->name] = $taxonomy->label;
							}

							$visibility_fld['widgetopts_visibility_taxonomies'] = array(
								'type'				=> 'widgetopts-select2',
								'label'				=> __('Taxonomies', 'widget-options'),
								'options'			=> $tax_array,
								'multi-select' 		=> true,
								'description' 		=> __('Click to search or select', 'widget-options')
							);
						}
					}

					if (isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['misc']) && '1' == $widget_options['settings']['visibility']['misc']) {
						$visibility_fld['widgetopts_visibility_misc'] = array(
							'type'				=> 'widgetopts-select2',
							'label'				=> __('Miscellaneous', 'widget-options'),
							'options'			=> array(
								'home'      =>  __('Home/Front', 'widget-options'),
								'blog'      =>  __('Blog', 'widget-options'),
								'archives'  =>  __('Archives', 'widget-options'),
								'404'       =>  __('404', 'widget-options'),
								'search'    =>  __('Search', 'widget-options')
							),
							'multi-select' 	=> true,
							'description' 	=> __('Click to search or select', 'widget-options')
						);
					}

					$sections['widgetopts-visibility'] = array(
						'fields' 	  =>  $visibility_fld
					);
				}

				if (
					isset($widget_options['logic']) && 'activate' == $widget_options['logic'] ||
					isset($widget_options['acf']) && 'activate' == $widget_options['acf']
				) {
					$settings_fld = array();

					if (isset($widget_options['logic']) && 'activate' == $widget_options['logic']) {
						$settings_fld['widgetopts_settings_logic'] = array(
							'type'          => 'textarea',
							'label' 		=> __('Display Logic', 'widget-options'),
							'description' 	=> __('<small>PLEASE NOTE that the display logic you introduce is EVAL\'d directly. Anyone who has access to edit widget appearance will have the right to add any code, including malicious and possibly destructive functions. There is an optional filter "widget_options_logic_override" which you can use to bypass the EVAL with your own code if needed.</small>', 'widget-options')
						);
					}

					//ACF
					if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) {
						$fields = array('' => __('Select Field', 'widget-options'));

						if (function_exists('acf_get_field_groups')) {
							$groups = acf_get_field_groups();
							if (is_array($groups)) {
								foreach ($groups as $group) {
									$fields_group = acf_get_fields($group);
									if (!empty($fields_group)) {
										foreach ($fields_group as $k => $fg) {
											$fields[$fg['key']] = $fg['label'];
										}
									}
								}
							}
						} else {
							$groups = apply_filters('acf/get_field_groups', array());
							if (is_array($groups)) {
								foreach ($groups as $group) {
									$fields_group = apply_filters('acf/field_group/get_fields', array(), $group['id']);
									if (!empty($fields_group)) {
										foreach ($fields_group as $k => $fg) {
											$fields[$fg['key']] = $fg['label'];
										}
									}
								}
							}
						}

						$settings_fld['widgetopts_acf_visibility'] = array(
							'type'          => 'select',
							'label' 		=> __('Show or Hide based on Advanced Custom Field', 'widget-options'),
							'options'       => array(
								'hide'      => __('Hide when Condition\'s Met', 'widget-options'),
								'show'      => __('Show when Condition\'s Met', 'widget-options')
							)
						);

						$settings_fld['widgetopts_acf_field'] = array(
							'type'				=> 'select',
							'label'				=> __('Select ACF Field', 'widget-options'),
							'options'			=> $fields
						);

						$settings_fld['widgetopts_acf_condition'] = array(
							'type'          => 'select',
							'label' 		=> __('Select Condition', 'widget-options'),
							'options'       => array(
								''      	 =>  __('Select Condition', 'widget-options'),
								'equal'      =>  __('Is Equal To', 'widget-options'),
								'not_equal'  =>  __('Is Not Equal To', 'widget-options'),
								'contains'   =>  __('Contains', 'widget-options'),
								'not_contains'   =>  __('Does Not Contain', 'widget-options'),
								'empty'      =>  __('Is Empty', 'widget-options'),
								'not_empty'  =>  __('Is Not Empty', 'widget-options')
							)
						);
						$settings_fld['widgetopts_acf_value'] = array(
							'type'          => 'textarea',
							'label' 		=> __('Conditional Value', 'widget-options'),
							'description' 	=> __('Add your Conditional Value here if you selected Equal to, Not Equal To or Contains on the selection above.', 'widget-options')
						);
					}

					$sections['widgetopts-settings'] = array(
						'fields' 	  =>  $settings_fld
					);
				}

				$upgrade_fld = array();
				$upgrade_fld['widgetopts_upgrade_section'] = array(
					'type' 	=> 'widgetopts-upgrade'
				);

				$sections['widgetopts-upgrade'] = array(
					'fields' 	  =>  $upgrade_fld
				);

				$form['widgetopts'] = array(
					'title' 	=>	__('Widget Options', 'widget-options'),
					'sections' 	=>  $sections
				);
			}

			return $form;
		}

		function fl_widgetopts_beaver_tabnav($name, $value, $field, $settings)
		{
			global $widget_options;
?>
			<div class="fl-builder-widgetopts-tab">
				<?php if (isset($widget_options['visibility']) && 'activate' == $widget_options['visibility']) { ?>
					<a onclick="widgetoptsBeaverModule.navClick(event)" href="#fl-builder-settings-section-widgetopts-visibility" class="widgetopts-s-active"><span class="dashicons dashicons-visibility"></span><?php _e('Visibility', 'widget-options'); ?></a>
				<?php } ?>
				<?php if (isset($widget_options['logic']) && 'activate' == $widget_options['logic'] && current_user_can('administrator')) { ?>
					<a onclick="widgetoptsBeaverModule.navClick(event)" href="#fl-builder-settings-section-widgetopts-settings" class=""><span class="dashicons dashicons-admin-generic"></span><?php _e('Settings', 'widget-options'); ?></a>
				<?php } ?>
				<a onclick="widgetoptsBeaverModule.navClick(event)" href="#fl-builder-settings-section-widgetopts-upgrade"><span class="dashicons dashicons-plus"></span><?php _e('More', 'widget-options'); ?></a>
			</div>
		<?php }

		function fl_widgetopts_beaver_scripts()
		{
			if (class_exists('FLBuilderModel') && FLBuilderModel::is_builder_active()) {
				$js_dir  = WIDGETOPTS_PLUGIN_URL . 'assets/js/';
				$css_dir = WIDGETOPTS_PLUGIN_URL . 'assets/css/';

				// Use minified libraries if SCRIPT_DEBUG is turned off
				$suffix  = (defined('SCRIPT_DEBUG') && SCRIPT_DEBUG) ? '' : '.min';

				wp_enqueue_style('widgetopts-beaver-css', $css_dir . 'beaver-widgetopts.css', array(), WIDGETOPTS_VERSION);
				wp_enqueue_style('widgetopts-beaver-select2-css', $css_dir . 'select2.min.css', array(), WIDGETOPTS_VERSION);
				wp_enqueue_style('widgetopts-jquery-ui', $css_dir . '/jqueryui/1.11.4/themes/ui-lightness/jquery-ui.css', array(), WIDGETOPTS_VERSION);
				wp_enqueue_style('jquery-ui');

				wp_enqueue_script(
					'beaver-widgetopts',
					$js_dir . 'jquery.widgetopts.beaver' . $suffix . '.js',
					array('jquery'),
					WIDGETOPTS_VERSION,
					true
				);
				wp_enqueue_script(
					'beaver-widgetopts-select2',
					$js_dir . 'select2.min.js',
					array('jquery', 'beaver-widgetopts'),
					WIDGETOPTS_VERSION,
					true
				);
				wp_enqueue_script(
					'beaver-widgetopts-s2',
					$js_dir . 'select2-settings' . $suffix . '.js',
					array('jquery', 'beaver-widgetopts'),
					WIDGETOPTS_VERSION,
					true
				);
			}
		}

		function fl_widgetopts_beaver_get_fld_options($settings, $field, $options = array())
		{
			if (!is_object($settings) && !is_array($settings)) return $options;

			foreach ($settings as $key => $val) {
				if ($key === $field) {
					if (is_array($val)) {
						foreach ($val as $v) {
							$options[$v] = $v;
						}
					} else {
						$options[$val] = $val;
					}
				} else {
					if (is_array($val) || is_object($val)) {
						$options = $this->fl_widgetopts_beaver_get_fld_options($val, $field, $options);
					}
				}
			}

			return array_unique($options);
		}

		function fl_widgetopts_beaver_select2($name, $value, $field, $settings)
		{
			$options = ($field['options']) ? $field['options'] : array();

			if (isset($field['options_from_field'])) {
				$options_field = $field['options_from_field'];
				$post_data = FLBuilderModel::get_post_data();
				$parent_settings = $post_data['node_settings'];

				$options = $this->fl_widgetopts_beaver_get_fld_options($parent_settings, $options_field, $options);
			}

			// Create attributes
			$attributes = '';
			if (isset($field['attributes']) && is_array($field['attributes'])) {
				foreach ($field['attributes'] as $key => $val) {
					$attributes .= $key . '="' . $val . '" ';
				}
			}

			if (!empty($options) && $value) {
				uksort($options, function ($key1, $key2) use ($value) {
					return array_search($key1, $value) <=> array_search($key2, $value);
				});
			}
			if (!isset($field['class'])) {
				$field['class'] = '';
			}

			// Show the select field
		?>
			<select name="<?php echo $name;
							if (isset($field['multi-select'])) echo '[]'; ?>" class="widgetopts-select2 <?php echo $field['class']; ?>" <?php if (isset($field['multi-select'])) echo 'multiple ';
																																		echo $attributes; ?> placeholder="<?php _e('Click to search or select', 'widget-options'); ?>">
				<option></option>
				<?php
				foreach ($options as $option_key => $option_val) :

					if (is_array($option_val) && isset($option_val['premium']) && $option_val['premium'] && true === FL_BUILDER_LITE) {
						continue;
					}

					$label = is_array($option_val) ? $option_val['label'] : $option_val;

					if (is_array($value) && in_array($option_key, $value)) {
						$selected = ' selected="selected"';
					} else if (!is_array($value) && selected($value, $option_key, true)) {
						$selected = ' selected="selected"';
					} else {
						$selected = '';
					}

				?>
					<option value="<?php echo $option_key; ?>" <?php echo $selected; ?>><?php echo $label; ?></option>
				<?php endforeach; ?>
			</select>
		<?php }

		function fl_widgetopts_upgrade($name, $value, $field, $settings)
		{ ?>
			<div class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-gopro">
				<p class="widgetopts-unlock-features">
					<span class="dashicons dashicons-lock"></span><?php _e('Unlock all Options', 'widget-options'); ?>
				</p>
				<p>
					<?php _e('Get the world\'s most complete widget management now with Beaver Builder integration! Upgrade to extended version to get:', 'widget-options'); ?>
				</p>
				<ul>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Animation Options', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Custom Styling Options', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Set Alignment per Devices', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('User Roles Visibility Restriction', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Fixed/Sticky Widget Options', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Days and Date Range Restriction', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Link Module Block Options', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Extended Taxonomy and Post Types Support', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Target URLs and Wildcard Restrictions', 'widget-options'); ?>
					</li>
					<li>
						<span class="dashicons dashicons-lock"></span> <?php _e('Beaver Builder Support', 'widget-options'); ?>
					</li>
				</ul>
				<p><strong><a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . '?utm_source=beaverbuilder&utm_medium=learnmore&utm_campaign=widgetoptsprotab'); ?>" class="button-primary" target="_blank"><?php _e('Learn More', 'widget-options'); ?></a></strong></p>
			</div>
			<?php }

		function widgetopts_beaver_is_node_visible($is_visible, $node)
		{

			//return if editing
			if (class_exists('FLBuilderModel') && FLBuilderModel::is_builder_active()) {
				return $is_visible;
			}

			global $widget_options;

			$settings 	= $node->settings;
			$hidden     = false;
			$visibility_opts    = isset($settings->widgetopts_visibility_show) ? $settings->widgetopts_visibility_show : 'hide';

			$tax_opts   = (isset($widget_options['settings']) && isset($widget_options['settings']['taxonomies_keys'])) ? $widget_options['settings']['taxonomies_keys'] : array();
			$is_misc    = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['misc'])) ? true : false;
			$is_types   = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['post_type'])) ? true : false;
			$is_tax     = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['taxonomies'])) ? true : false;
			$is_inherit = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['inherit'])) ? true : false;

			// echo '<pre>';
			// print_r( $settings );
			// echo '</pre>';

			//pages
			if ($is_misc && ((is_home() && is_front_page()) || is_front_page())) {
				if (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && in_array('home', $settings->widgetopts_visibility_misc) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings->widgetopts_visibility_misc) || (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && !in_array('home', $settings->widgetopts_visibility_misc))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_home', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_home()) {
				//NOT CHECKED YET
				if (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && in_array('blog', $settings->widgetopts_visibility_misc) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings->widgetopts_visibility_misc) || (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && !in_array('blog', $settings->widgetopts_visibility_misc))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_blog', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_tax && is_tag()) {

				if (!isset($settings->widgetopts_visibility_taxonomies) || (isset($settings->widgetopts_visibility_taxonomies) && !is_array($settings->widgetopts_visibility_taxonomies))) {
					$settings->widgetopts_visibility_taxonomies = array();
				}

				if (in_array('post_tag', $settings->widgetopts_visibility_taxonomies) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all tags
				} elseif (in_array('post_tag', $settings->widgetopts_visibility_taxonomies) && $visibility_opts == 'show') {
					$hidden = false; //hide to all tags
				}
				//
				// //do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_tags', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_tax && is_tax()) {
				$term = get_queried_object();
				//taxonomies page
				if (!isset($settings->widgetopts_visibility_taxonomies) || (isset($settings->widgetopts_visibility_taxonomies) && !is_array($settings->widgetopts_visibility_taxonomies))) {
					$settings->widgetopts_visibility_taxonomies = array();
				}
				// print_r( $term_lists );
				if (in_array($term->taxonomy, $settings->widgetopts_visibility_taxonomies) && $visibility_opts == 'hide') {
					$hidden = true; //hide to all tags
				} elseif (!in_array($term->taxonomy, $settings->widgetopts_visibility_taxonomies) && $visibility_opts == 'show') {
					$hidden = true; //hide to all tags
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_taxonomies', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_archive()) {
				//archives page
				if (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && in_array('archives', $settings->widgetopts_visibility_misc) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings->widgetopts_visibility_misc) || (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && !in_array('archives', $settings->widgetopts_visibility_misc))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_archives', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_404()) {
				//404 page
				if (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && in_array('404', $settings->widgetopts_visibility_misc) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings->widgetopts_visibility_misc) || (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && !in_array('404', $settings->widgetopts_visibility_misc))) && $visibility_opts == 'show') {
					$hidden = true; //hide if not checked on visible pages
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_404', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif ($is_misc && is_search()) {
				if (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && in_array('search', $settings->widgetopts_visibility_misc) && $visibility_opts == 'hide') {
					$hidden = true; //hide if checked on hidden pages
				} elseif ((!isset($settings->widgetopts_visibility_misc) || (isset($settings->widgetopts_visibility_misc) && is_array($settings->widgetopts_visibility_misc) && !in_array('search', $settings->widgetopts_visibility_misc))) && $visibility_opts == 'show') {
					$hidden = true;
				}

				//do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_search', $hidden);
				if ($hidden) {
					return false;
				}
			} elseif (is_single() && !is_page()) {
				global $wp_query;
				$post = $wp_query->post;

				if (!isset($settings->widgetopts_visibility_types) || ($is_types && !isset($settings->widgetopts_visibility_types)) || !is_array($settings->widgetopts_visibility_types)) {
					$settings->widgetopts_visibility_types = array();
				}

				if ($visibility_opts == 'hide' && in_array($post->post_type, $settings->widgetopts_visibility_types)) {
					$hidden = true; //hide if exists on hidden pages
				} elseif ($visibility_opts == 'show' && !in_array($post->post_type, $settings->widgetopts_visibility_types)) {
					$hidden = true; //hide if doesn't exists on visible pages
				}

				// do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_single', $hidden);


				// $taxonomy_names  = get_post_taxonomies( $post->ID );
				// $array_intersect = array_intersect( $tax_opts, $taxonomy_names );

				if (!isset($settings->widgetopts_visibility_tax_category)) {
					$settings->widgetopts_visibility_tax_category = array();
				}

				if (isset($settings->widgetopts_visibility_tax_category) && !empty($settings->widgetopts_visibility_tax_category)) {
					$cats	= wp_get_post_categories($post->ID);

					if (is_array($cats) && !empty($cats)) {
						$checked_cats   = $settings->widgetopts_visibility_tax_category;
						$intersect      = array_intersect($cats, $checked_cats);
						if (!empty($intersect) && $visibility_opts == 'hide') {
							$hidden = true;
						} elseif (!empty($intersect) && $visibility_opts == 'show') {
							$hidden = false;
						}

						$hidden = apply_filters('widgetopts_beaver_visibility_single_category', $hidden);
					}
				}

				if ($hidden) {
					return false;
				}
			} elseif ($is_types && is_page()) {
				global $wp_query;

				$post = $wp_query->post;

				//do post type condition first
				if (isset($settings->widgetopts_visibility_types) && is_array($settings->widgetopts_visibility_types) && in_array('page', $settings->widgetopts_visibility_types)) {

					if (!is_array($settings->widgetopts_visibility_types)) {
						$settings->widgetopts_visibility_types = array();
					}

					if ($visibility_opts == 'hide' && in_array('page', $settings->widgetopts_visibility_types)) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !in_array('page', $settings->widgetopts_visibility_types)) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				} else {
					// print_r( $settings['widgetopts_pages'] );
					//do per pages condition
					if (!isset($settings->widgetopts_visibility_pages) || (isset($settings->widgetopts_visibility_pages) && !is_array($settings->widgetopts_visibility_pages))) {
						$settings->widgetopts_visibility_pages = array();
					}

					if ($visibility_opts == 'hide' && in_array($post->ID, $settings->widgetopts_visibility_pages)) {
						$hidden = true; //hide if exists on hidden pages
					} elseif ($visibility_opts == 'show' && !in_array($post->ID, $settings->widgetopts_visibility_pages)) {
						$hidden = true; //hide if doesn't exists on visible pages
					}
				}

				// //do return to bypass other conditions
				$hidden = apply_filters('widgetopts_beaver_visibility_page', $hidden);
				if ($hidden) {
					return false;
				}
			}

			//ACF
			if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) {
				if (isset($settings->widgetopts_acf_field) && !empty($settings->widgetopts_acf_field)) {
					$acf = get_field_object($settings->widgetopts_acf_field);
					if ($acf && is_array($acf)) {
						$acf_visibility    = isset($settings->widgetopts_acf_visibility) ? $settings->widgetopts_acf_visibility : 'hide';

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

						switch ($settings->widgetopts_acf_condition) {
							case 'equal':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && $acf['value'] == $settings->widgetopts_acf_value) {
										$hidden = false;
									} else if ('show' == $acf_visibility && $acf['value'] != $settings->widgetopts_acf_value) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && $acf['value'] == $settings->widgetopts_acf_value) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && $acf['value'] != $settings->widgetopts_acf_value) {
										$hidden = false;
									}
								}
								break;

							case 'not_equal':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && $acf['value'] == $settings->widgetopts_acf_value) {
										$hidden = true;
									} else if ('show' == $acf_visibility && $acf['value'] != $settings->widgetopts_acf_value) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && $acf['value'] == $settings->widgetopts_acf_value) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && $acf['value'] != $settings->widgetopts_acf_value) {
										$hidden = true;
									}
								}
								break;

							case 'contains':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) !== false) {
										$hidden = false;
									} else if ('show' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) === false) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) !== false) {
										$hidden = true;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) === false) {
										$hidden = false;
									}
								}
								break;

							case 'not_contains':
								if (isset($acf['value'])) {
									if ('show' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) !== false) {
										$hidden = true;
									} else if ('show' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) === false) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) !== false) {
										$hidden = false;
									} else if ('hide' == $acf_visibility && strpos($acf['value'], $settings->widgetopts_acf_value) === false) {
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
						$hidden = apply_filters('widgetopts_beaver_visibility_acf', $hidden);
						if ($hidden) {
							return false;
						}
					}
				}
			}

			//widget logic
			if (isset($widget_options['logic']) && 'activate' == $widget_options['logic']) {
				if (isset($settings->widgetopts_settings_logic) && !empty($settings->widgetopts_settings_logic)) {
					//do widget logic
					$display_logic = stripslashes(trim($settings->widgetopts_settings_logic));
					$display_logic = apply_filters('widget_options_logic_override', $display_logic);
					$display_logic = apply_filters('extended_widget_options_logic_override', $display_logic);
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
					try {
						if (!widgetopts_safe_eval($display_logic)) {
							return false;
						}
					} catch (ParseError $e) {
						return false;
					}
				}
			}

			return $is_visible;
		}
		function widgetopts_plugin_check()
		{
			if (!defined('WIDGETOPTS_PLUGIN_NAME')) { ?>
				<div class="widgetopts_activated_notice notice-error notice" style="box-shadow: 0 1px 1px 0 rgba(0,0,0,.1);">
					<p>
						<?php _e('<strong>Widget Options Plugin</strong> is required for the <em>Widget Options for Beaver Builder</em> to work properly. Please get the plugin <a href="https://wordpress.org/plugins/widget-options/" target="_blank">here</a>. Thanks!', 'widget-options'); ?>
					</p>
				</div>
<?php }
		}
	}

	add_action('plugins_loaded', array('WP_Widget_Options_Beaver', 'init'));
// new WP_Widget_Options_Beaver();

endif;
