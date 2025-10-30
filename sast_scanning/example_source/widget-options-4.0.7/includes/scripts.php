<?php

/**
 * Scripts
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Load Scripts
 *
 * Enqueues the required scripts.
 *
 * @since 3.0
 * @return void
 */

function widgetopts_load_scripts()
{
      global $pagenow;
      $css_dir = WIDGETOPTS_PLUGIN_URL . 'assets/css/';
      wp_enqueue_style('widgetopts-styles', $css_dir . 'widget-options.css', array(), WIDGETOPTS_VERSION);

      if (isset($pagenow) && $pagenow === 'customize.php') {
            wp_add_inline_script(
                  'customize-controls',
                  '(function(){
                        document.querySelector("#save").onclick = function() {
                              setTimeout(function(){wp.customize.previewer.refresh();}, 3000);
                        }
                  })();',
                  'after'
            );
      }
}
add_action('wp_enqueue_scripts', 'widgetopts_load_scripts');
add_action('customize_controls_enqueue_scripts', 'widgetopts_load_scripts');

/**
 * Load Admin Scripts
 *
 * Enqueues the required admin scripts.
 *
 * @since 3.0
 * @global $widget_options
 * @param string $hook Page hook
 * @return void
 */
if (!function_exists('widgetopts_load_admin_scripts')) :
      function widgetopts_load_admin_scripts($hook)
      {
            global $widget_options, $wp_version;

            $in_footer_args = false;
            $is_6_3_and_above = version_compare($wp_version, '6.3', '>=');
            if ($is_6_3_and_above) {
                  $in_footer_args = array(
                        'in_footer' => true,
                        'strategy'  => 'defer',
                  );
            }

            $js_dir  = WIDGETOPTS_PLUGIN_URL . 'assets/js/';
            $css_dir = WIDGETOPTS_PLUGIN_URL . 'assets/css/';
            $is_siteorigin  = (isset($widget_options['siteorigin'])) ? $widget_options['siteorigin'] : '';

            // Use minified libraries if SCRIPT_DEBUG is turned off
            $suffix  = (defined('SCRIPT_DEBUG') && SCRIPT_DEBUG) ? '' : '.min';

            wp_enqueue_style('widgetopts-admin-styles', $css_dir . 'admin.css', array(), WIDGETOPTS_VERSION);

            wp_enqueue_script(
                  'widgetopts-global-script',
                  $js_dir . 'widgetopts.global.js',
                  array('jquery'),
                  WIDGETOPTS_VERSION,
                  ($is_6_3_and_above ? $in_footer_args : true)
            );

            //load only on admin pages with widgets
            if (($is_siteorigin) || (!$is_siteorigin && in_array($hook, apply_filters('widgetopts_load_option-tabs_scripts', array('widgets.php', 'customize.php', 'nav-menus.php'))))) {

                  if (!in_array($hook, apply_filters('widgetopts_exclude_jqueryui', array('toplevel_page_et_divi_options', 'toplevel_page_wpcf7', 'edit.php')))) {
                        wp_enqueue_style('jquery-ui');
                  }

                  if (in_array($hook, apply_filters('widgetopts_load_liveFilter_scripts', array('widgets.php', 'nav-menus.php')))) {
                        wp_enqueue_script(
                              'jquery-liveFilter',
                              plugins_url('assets/js/jquery.liveFilter.js', dirname(__FILE__)),
                              array('jquery'),
                              WIDGETOPTS_VERSION,
                              ($is_6_3_and_above ? $in_footer_args : true)
                        );
                  }

                  wp_enqueue_script(
                        'jquery-widgetopts-option-tabs',
                        plugins_url('assets/js/wpWidgetOpts.js', dirname(__FILE__)),
                        array('jquery', 'jquery-ui-core', 'jquery-ui-tabs', 'jquery-ui-datepicker'),
                        WIDGETOPTS_VERSION,
                        ($is_6_3_and_above ? $in_footer_args : true)
                  );


                  wp_enqueue_style('jquery-widgetopts-select2-css', plugins_url('assets/css/select2.min.css', dirname(__FILE__)), array(), WIDGETOPTS_VERSION);

                  if (!wp_script_is('select2', 'enqueued')) {
                        wp_enqueue_script(
                              'jquery-widgetopts-select2-script',
                              $js_dir . 'select2.min.js',
                              array('jquery'),
                              WIDGETOPTS_VERSION,
                              ($is_6_3_and_above ? $in_footer_args : true)
                        );
                  }

                  wp_enqueue_style('jquery-widgetopts-multiselect-css', plugins_url('assets/css/bootstrap-multiselect.min.css', dirname(__FILE__)), array(), WIDGETOPTS_VERSION);
                  wp_enqueue_script(
                        'jquery-widgetopts-multiselect-script',
                        $js_dir . 'bootstrap-multiselect.min.js',
                        array('jquery'),
                        WIDGETOPTS_VERSION,
                        ($is_6_3_and_above ? $in_footer_args : true)
                  );

                  $form = '<div id="widgetopts-widgets-chooser">
                  <label class="screen-reader-text" for="widgetopts-search-chooser">' . __('Search Sidebar', 'widget-options') . '</label>
                  <input type="text" id="widgetopts-search-chooser" class="widgetopts-widgets-search" placeholder="' . __('Search sidebar&hellip;', 'widget-options') . '" />
                  <div class="widgetopts-search-icon" aria-hidden="true"></div>
                  <button type="button" class="widgetopts-clear-results"><span class="screen-reader-text">' . __('Clear Results', 'widget-options') . '</span></button>
                  <p class="screen-reader-text" id="widgetopts-chooser-desc">' . __('The search results will be updated as you type.', 'widget-options') . '</p>
              </div>';

                  $btn_controls = '';
                  if (isset($widget_options['move']) && 'activate' == $widget_options['move']) {
                        $btn_controls .= ' | <button type="button" class="button-link widgetopts-control" data-action="move">' . __('Move', 'widget-options') . '</button>';
                  }

                  $sidebaropts = '';
                  if (isset($widget_options['widget_area']) && 'activate' == $widget_options['widget_area']) {
                        /* Updated by Haive Vistal - 04/20/2023 - Make sure no empty space in under the widgets if no activated links */
                        $remove_widget_link = 0;
                        $download_backup_link = 0;
                        $delete_all_widget_link = 0;

                        if (isset($widget_options['settings']['widget_area']) && isset($widget_options['settings']['widget_area']['remove']) && '1' == $widget_options['settings']['widget_area']['remove']) {
                              $remove_widget_link = 1;
                        }
                        if (isset($widget_options['settings']['widget_area']) && isset($widget_options['settings']['widget_area']['backup']) && '1' == $widget_options['settings']['widget_area']['backup']) {
                              $download_backup_link = 1;
                        }

                        if (isset($widget_options['settings']['widget_area']) && isset($widget_options['settings']['widget_area']['remove']) && '1' == $widget_options['settings']['widget_area']['remove']) {
                              $delete_all_widget_link = 1;
                        }

                        if ($remove_widget_link == 1 || $download_backup_link == 1 || $delete_all_widget_link == 1) {
                              $sidebaropts = '<div class="widgetopts-sidebaropts">';
                              if ($remove_widget_link == 1) {
                                    $sidebaropts .= '<a href="#" class="sidebaropts-clear">
                            <span class="dashicons dashicons-warning"></span> ' . __('Remove All Widgets', 'widget-options') . '
                        </a>';
                              }
                              if ($download_backup_link == 1) {
                                    $sidebaropts .= '<a href="' . esc_url(wp_nonce_url(admin_url('tools.php?page=widgetopts_migrator_settings&action=export&single_sidebar=__sidebaropts__'), 'widgeopts_export', 'widgeopts_nonce_export')) . '">
                            <span class="dashicons dashicons-download"></span> ' . __('Download Backup', 'widget-options') . '
                        </a>';
                              }
                              if ($delete_all_widget_link == 1) {
                                    $sidebaropts .= '<div class="sidebaropts-confirm"><p>
                          ' . __('Are you sure you want to DELETE ALL widgets associated to __sidebar_opts__?', 'widget-options') . '
                          </p>
                          <button class="button">' . __('No', 'widget-options') . '</button>
                          <button class="button button-primary">' . __('Yes', 'widget-options') . '</button>
                        </div>';
                              }
                              $sidebaropts .= '</div>';
                        }
                  }

                  /* Added by Haive Vistal - 04/20/2023 - Default link for all widgets to go through widget options panel settings */
                  // $sidebaropts .= '<div class="widgetopts-super  widgetopts-sidebaropts">';
                  //     $sidebaropts .= '<a href="'. esc_url( wp_nonce_url( admin_url('options-general.php?page=widgetopts_plugin_settings'), 'widgeopts_setings', 'widgeopts_nonce_settings') ) .'">
                  //         <span class="dashicons dashicons-admin-settings"></span> '. __( 'Enable more Widget Options superpowers', 'widget-options' ) .'
                  //       </a>';
                  // $sidebaropts .= '</div>';

                  wp_localize_script('jquery-widgetopts-option-tabs', 'widgetopts10n', array('ajax_url' => admin_url('admin-ajax.php'), 'opts_page' => esc_url(admin_url('options-general.php?page=widgetopts_plugin_settings')), 'search_form' => $form, 'sidebaropts' => $sidebaropts, 'controls' => $btn_controls, 'translation' => array('manage_settings' => __('Manage Widget Options', 'widget-options'), 'search_chooser' => __('Search sidebar&hellip;', 'widget-options'))));
            } else {
                  wp_localize_script('widgetopts-global-script', 'widgetopts10n', array('ajax_url' => admin_url('admin-ajax.php'), 'opts_page' => esc_url(admin_url('options-general.php?page=widgetopts_plugin_settings')), 'translation' => array('manage_settings' => __('Manage Widget Options', 'widget-options'), 'search_chooser' => __('Search sidebar&hellip;', 'widget-options'))));
            }

            if (in_array($hook, apply_filters('widgetopts_load_settings_scripts', array('settings_page_widgetopts_plugin_settings')))) {
                  wp_register_script(
                        'jquery-widgetopts-settings',
                        $js_dir . 'settings' . $suffix . '.js',
                        array('jquery'),
                        WIDGETOPTS_VERSION,
                        ($is_6_3_and_above ? $in_footer_args : true)
                  );

                  $translation = array(
                        'save_settings'         => __('Save Settings', 'widget-options'),
                        'close_settings'        => __('Close', 'widget-options'),
                        'show_settings'         => __('Configure Settings', 'widget-options'),
                        'hide_settings'         => __('Hide Settings', 'widget-options'),
                        'show_description'      => __('Learn More', 'widget-options'),
                        'hide_description'      => __('Hide Details', 'widget-options'),
                        'show_information'      => __('Show Details', 'widget-options'),
                        'activate'              => __('Enable', 'widget-options'),
                        'deactivate'            => __('Disable', 'widget-options'),
                        'successful_save'       => __('Settings saved successfully for %1$s.', 'widget-options'),
                        'deactivate_btn'        => __('Deactivate License', 'widget-options'),
                        'activate_btn'          => __('Activate License', 'widget-options'),
                        'status_valid'             => __('Valid', 'widget-options'),
                        'status_invalid'        => __('Invalid', 'widget-options'),
                  );

                  wp_enqueue_script('jquery-widgetopts-settings');
                  wp_localize_script('jquery-widgetopts-settings', 'widgetopts', array('translation' => $translation, 'ajax_action' => 'widgetopts_ajax_settings', 'ajax_nonce' => wp_create_nonce('widgetopts-settings-nonce'),));
            }
      }
      add_action('admin_enqueue_scripts', 'widgetopts_load_admin_scripts', 100);
endif;

if (!function_exists('widgetopts_widgets_footer')) {
      function widgetopts_widgets_footer()
      {
            global $widget_options; ?>
            <div class="widgetsopts-chooser" style="display:none;">
                  <?php if (isset($widget_options['search']) && 'activate' == $widget_options['search']) : ?>
                        <div id="widgetopts-widgets-chooser">
                              <label class="screen-reader-text" for="widgetopts-search-chooser"><?php _e('Search Sidebar', 'widget-options'); ?></label>
                              <input type="text" id="widgetsopts-widgets-search" class="widgetopts-widgets-search widgetsopts-widgets-search" placeholder="Search sidebar…">
                              <div class="widgetopts-search-icon" aria-hidden="true"></div>
                              <button type="button" class="widgetopts-clear-results"><span class="screen-reader-text"><?php _e('Clear Results', 'widget-options'); ?></span></button>
                              <p class="screen-reader-text" id="widgetopts-chooser-desc"><?php _e('The search results will be updated as you type.', 'widget-options'); ?></p>
                        </div>
                  <?php endif; ?>
                  <ul class="widgetopts-chooser-sidebars"></ul>
                  <div class="widgetsopts-chooser-actions">
                        <button class="button widgetsopts-chooser-cancel"><?php _e('Cancel', 'widget-options'); ?></button>
                        <button class="button button-primary widgetopts-chooser-action"><span><?php _e('Move', 'widget-options'); ?></span> <?php _e('Widget', 'widget-options'); ?></button>
                  </div>
            </div>
      <?php }
      add_action('admin_footer-widgets.php', 'widgetopts_widgets_footer');
}

if (!function_exists('widgetopts_widgets_footer_additional_script')) {
      function widgetopts_widgets_footer_additional_script()
      {
      ?>
            <script>
                  (function() {
                        /*widget option search option function*/
                        function widgetopts_seach_button_function(e) {
                              let current_parent = e.target.closest('.extended-widget-opts-parent-option');
                              let current_select = current_parent.querySelector('.extended-widget-opts-select2-dropdown');
                              current_select.classList.add('select2-hidden-accessible');
                              current_parent.querySelector('.select2-container').classList.remove('widgetopts-is-hidden');

                              jQuery(current_parent).find('.multiselect-native-select .btn-group').addClass('hide');
                              jQuery(this).css({
                                    "background-color": "#3D434A",
                                    color: "#fff"
                              });

                              jQuery(this).parent().find('.widgetopts-dropdown-option-btn').css({
                                    "background-color": "#fff",
                                    color: "#777A80"
                              });
                        }

                        jQuery(document).on('click', '.widgetopts-search-option-btn', widgetopts_seach_button_function);

                        // let search_btns = document.getElementsByClassName('widgetopts-search-option-btn');
                        // for (let i = 0; i < search_btns.length; i++) {
                        //       search_btns[i].addEventListener('click', widgetopts_seach_button_function, false);
                        // } /*end of widget option search option function*/

                        /*widget option dropdown option function*/
                        function widgetopts_dropdown_button_function(e) {
                              let current_parent = e.target.closest('.extended-widget-opts-parent-option');
                              let current_select = current_parent.querySelector('.extended-widget-opts-select2-dropdown');
                              current_select.classList.remove('select2-hidden-accessible');
                              current_parent.querySelector('.select2-container').classList.add('widgetopts-is-hidden');

                              jQuery(this).css({
                                    "background-color": "#3D434A",
                                    color: "#fff"
                              });

                              jQuery(this).parent().find('.widgetopts-search-option-btn').css({
                                    "background-color": "#fff",
                                    color: "#777A80"
                              });

                              if (jQuery(current_parent).find('.multiselect-native-select').length > 0) {
                                    jQuery(current_parent).find('.multiselect-native-select .btn-group').removeClass('hide').find('.multiselect-container.dropdown-menu').removeClass('show');
                              } else {
                                    let spinner = '<span class="spinner multiple-spinner" style="visibility: visible; float: none; display: inline-block; vertical-align: bottom;"></span>';
                                    jQuery(e.target).parent().append(spinner);
                                    jQuery(current_select).multiselect({
                                          onInitialized: function(select, container) {
                                                setTimeout(function() {
                                                      jQuery('.multiple-spinner').remove();
                                                }, 500);
                                          },
                                          onChange: function(option, checked, select) {
                                                jQuery(option).parents('.extended-widget-opts-parent-option').children('div:nth-child(1)').append(spinner);
                                                setTimeout(function() {
                                                      jQuery('.multiple-spinner').remove();
                                                }, 500);
                                          },
                                          buttonText: () => 'Click to add more',
                                          keepOrder: true
                                    });
                              }
                        }

                        jQuery(document).on('click', '.widgetopts-dropdown-option-btn', widgetopts_dropdown_button_function);

                        // let drop_btns = document.getElementsByClassName('widgetopts-dropdown-option-btn');
                        // for (let i = 0; i < drop_btns.length; i++) {
                        //       drop_btns[i].addEventListener('click', widgetopts_dropdown_button_function, false);
                        // } /*end of widget option dropdown option function*/

                        jQuery(document).on('click', '.multiselect-native-select .multiselect.dropdown-toggle', function() {
                              jQuery(this).parent().find('.multiselect-container.dropdown-menu').each(function() {
                                    if (jQuery(this).hasClass('show')) {
                                          jQuery(this).removeClass('show');
                                    } else {
                                          jQuery(this).addClass('show');
                                    }
                              });

                        });
                  })();
            </script>
<?php }
      add_action('admin_footer-widgets.php', 'widgetopts_widgets_footer_additional_script', 999);
}
?>